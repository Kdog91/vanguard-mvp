"""
Security controls (update 26f), following the Vanguard Analytics Security Outline.

- Web-address imports (SSRF): only http/https, every address the name resolves to must be public, every
  redirect is checked again, and the socket's real peer address is checked after connecting, before any
  request is sent (stops DNS-rebinding tricks).
- Database connections on the public app: private, loopback and cloud-metadata hosts are refused, and
  SQLite file paths (files on the server) are refused.
- Upload limits: file type, size, rows and columns.
- Safe downloads: text cells that start with = + - @ (or tab / carriage return) are prefixed with an
  apostrophe so spreadsheet programs show them as text instead of running them as formulas.
- Temporary files: everything the app writes is named vanguard_*; files older than a few hours are deleted.
- Audit log: one JSON line per event (sign-in, load, export, error) with the account as a short one-way
  code, file type and size only. Never data, passwords or full web addresses.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
import urllib.request
from urllib.parse import urlparse

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ON_CLOUD = HERE.replace("\\", "/").startswith("/mount/src")

# ------------------------------------------------------------------ limits
MAX_UPLOAD_MB = 200          # matches server.maxUploadSize in .streamlit/config.toml
MAX_ROWS = 1_000_000         # in-memory analysis; larger files use large-file mode on your own computer
MAX_COLUMNS = 2_000
MAX_CELLS = 50_000_000
ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet", ".orc")
TEMP_PREFIX = "vanguard_"
TEMP_MAX_AGE_HOURS = 3


class SecurityError(ValueError):
    """A request the app refuses for safety reasons. The message is safe to show to the user."""


def check_upload(name: str, size_bytes: int | None) -> None:
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SecurityError(f"Files of type '{ext or 'none'}' are not accepted. Allowed: "
                            + ", ".join(e.lstrip(".") for e in ALLOWED_EXTENSIONS) + ".")
    if size_bytes is not None and size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise SecurityError(f"That file is larger than {MAX_UPLOAD_MB} MB.")


def check_frame(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    if cols > MAX_COLUMNS:
        raise SecurityError(f"The file has {cols:,} columns; the limit is {MAX_COLUMNS:,}.")
    if rows > MAX_ROWS or rows * max(cols, 1) > MAX_CELLS:
        raise SecurityError(f"The file has {rows:,} rows × {cols:,} columns, more than the app can analyze in memory "
                            f"(limit {MAX_ROWS:,} rows). Use large-file mode on your own computer, or load a sample.")


# ------------------------------------------------------------------ addresses
def _bad_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return not a.is_global or a.is_multicast or a.is_reserved or a.is_loopback or a.is_link_local or a.is_private


def check_host(host: str, port: int | None = None) -> None:
    """Refuse a host name or address that is (or resolves to) anything other than a public internet address."""
    host = (host or "").strip().strip("[]")
    if not host:
        raise SecurityError("The address has no host name.")
    if host.lower() in ("localhost", "metadata", "metadata.google.internal") or host.lower().endswith(
            (".localhost", ".local", ".internal")):
        raise SecurityError("That address points to a private or internal network, which is not allowed.")
    try:
        infos = socket.getaddrinfo(host, port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise SecurityError(f"The host name '{host}' could not be found.")
    if not infos or any(_bad_ip(i[4][0]) for i in infos):
        raise SecurityError("That address points to a private or internal network, which is not allowed.")


def check_url(url: str) -> str:
    url = (url or "").strip()
    p = urlparse(url)
    if p.scheme.lower() not in ("http", "https"):
        raise SecurityError("The address must start with http:// or https://")
    if p.username or p.password:
        raise SecurityError("Addresses with a user name or password in them are not allowed.")
    check_host(p.hostname or "", p.port)
    return url


def _check_peer(sock) -> None:
    ip = sock.getpeername()[0]
    if _bad_ip(ip):
        sock.close()
        raise SecurityError("That address points to a private or internal network, which is not allowed.")


def _guarded_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *a, **kw):
    """socket.create_connection, then refuse the socket if its real peer is private (checked before any
    TLS handshake or HTTP request is sent)."""
    sock = socket.create_connection(address, timeout, source_address, *a, **kw)
    _check_peer(sock)
    return sock


class _HTTPConn(http.client.HTTPConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _guarded_connection


class _HTTPSConn(http.client.HTTPSConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _guarded_connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConn, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConn, req, context=self._context)


class _Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_fetch(url: str, max_bytes: int, timeout: int = 30) -> tuple[bytes, str, str]:
    """Download a public web address. Returns (data, content type, final address)."""
    check_url(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _HTTPHandler(), _HTTPSHandler(),
                                         _Redirects())
    req = urllib.request.Request(url, headers={"User-Agent": "VanguardAnalytics/1.0"})
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
        return data, (resp.headers.get("Content-Type") or "").lower(), resp.geturl()


def check_database(url) -> None:
    """On the public app, only internet-reachable database servers; no files on the server."""
    if not ON_CLOUD:
        return                       # your own computer: localhost and private networks are yours
    if url.drivername.startswith("sqlite"):
        raise SecurityError("SQLite files can only be opened when the app runs on your own computer. On the public "
                            "app, upload the table as a file or use a database server reachable from the internet.")
    check_host(url.host or "", url.port)


# ------------------------------------------------------------------ downloads
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _escape_cell(v):
    if isinstance(v, str) and v.startswith(_FORMULA_START):
        try:
            float(v.replace(",", ""))          # plain negative numbers such as "-12.5" are left alone
            return v
        except ValueError:
            return "'" + v
    return v


def escape_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df that is safe to open in Excel or Google Sheets (CSV injection protection)."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = s.astype(object).map(_escape_cell)
    out.columns = [_escape_cell(str(c)) for c in out.columns]
    return out


def csv_bytes(df: pd.DataFrame, **kw) -> bytes:
    kw.setdefault("index", False)
    return escape_formulas(df).to_csv(**kw).encode("utf-8")


def excel_bytes(df: pd.DataFrame) -> bytes:
    import io
    buf = io.BytesIO()
    escape_formulas(df).to_excel(buf, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ temporary files
_last_sweep = [0.0]


def temp_dir() -> str:
    return tempfile.mkdtemp(prefix=TEMP_PREFIX)


def sweep_temp(max_age_hours: float = TEMP_MAX_AGE_HOURS, force: bool = False) -> int:
    """Delete vanguard_* temporary files and folders older than max_age_hours. Runs at most every 10 minutes."""
    now = time.time()
    if not force and now - _last_sweep[0] < 600:
        return 0
    _last_sweep[0] = now
    root, removed = tempfile.gettempdir(), 0
    for name in os.listdir(root):
        if not name.startswith(TEMP_PREFIX):
            continue
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) < max_age_hours * 3600:
                continue
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


# ------------------------------------------------------------------ audit log
_log = logging.getLogger("vanguard.audit")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def account_code(email: str) -> str:
    """Short one-way code for an account, so logs can be grouped by user without storing the email."""
    return hashlib.sha256((email or "anonymous").lower().encode()).hexdigest()[:12] if email else "anonymous"


def audit(event: str, **fields) -> None:
    try:
        import auth
        user = account_code(auth.email())
    except Exception:
        user = "anonymous"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "user": user}
    rec.update({k: v for k, v in fields.items() if v is not None})
    _log.info("AUDIT " + json.dumps(rec, default=str))


def host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""
