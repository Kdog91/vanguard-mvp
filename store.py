"""
Saved projects (update 27), stored in Supabase.

Design (Vanguard Analytics Security Outline, "Saved projects"):
- Only the app's server talks to Supabase, with the secret key kept in Streamlit Secrets. The projects
  table has row-level security on and no policies, and is not exposed to the public (anon) key, so the
  browser and the public key can read nothing.
- Each account's projects are separated by an owner code: a one-way hash of the sign-in provider's
  permanent account id (not the email). Every query filters on it.
- The data itself is encrypted by the app (Fernet: AES-128-CBC + HMAC-SHA256) with a key Supabase never
  sees, then stored as a file in a private Storage bucket. Supabase also encrypts its disks.
- Customers can delete one project or all of them. Projects not opened for RETENTION_DAYS are deleted
  automatically.

Secrets (Streamlit Cloud: app -> Settings -> Secrets):
    [supabase]
    url = "https://YOUR-PROJECT.supabase.co"
    secret_key = "sb_secret_..."          # or the legacy service_role key
    encryption_key = "...44 characters..."  # py -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import time
import uuid

import pandas as pd

TABLE = "projects"
BUCKET = "projects"
RETENTION_DAYS = 180
MAX_PROJECTS = 25
MAX_STORED_MB = 50
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COLUMNS = "id,name,source,n_rows,n_cols,size_bytes,created_at,updated_at,expires_at"


class StoreError(RuntimeError):
    """A problem the user can be told about in plain words."""


def owner_code(subject: str) -> str:
    if not subject:
        raise StoreError("Sign in to save projects.")
    return hashlib.sha256(("vanguard-owner:" + subject).encode()).hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


class SupabaseStore:
    def __init__(self, url: str, secret_key: str, encryption_key: str, client=None):
        from cryptography.fernet import Fernet
        url = (url or "").strip().rstrip("/")
        if not re.match(r"^https://[a-z0-9-]+\.supabase\.(co|in)$", url) and not url.startswith("http://127.0.0.1"):
            raise StoreError("The Supabase url in Secrets should look like https://abcd1234.supabase.co")
        if not secret_key:
            raise StoreError("The Supabase secret_key is missing from Secrets.")
        if secret_key.startswith("sb_publishable_"):
            raise StoreError("That is the publishable key. Saved projects need the secret key (sb_secret_...).")
        try:
            self._fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        except Exception:
            raise StoreError("The encryption_key in Secrets is not valid. Make one with the command in store.py.")
        self.url = url
        self._key = secret_key.strip()
        import httpx
        self._http = client or httpx.Client(timeout=30)
        self._last_cleanup = 0.0

    # ------------------------------------------------------------------ http
    def _headers(self, extra: dict | None = None) -> dict:
        h = {"apikey": self._key}
        if self._key.startswith("eyJ"):                 # legacy service_role key is a JWT
            h["Authorization"] = f"Bearer {self._key}"
        if extra:
            h.update(extra)
        return h

    def _call(self, method: str, path: str, **kw):
        extra = kw.pop("headers", None) or {}
        try:
            r = self._http.request(method, self.url + path, headers=self._headers(extra), **kw)
            if r.status_code in (400, 401, 403) and "Authorization" not in self._headers() \
                    and path.startswith("/storage/"):
                # some Storage versions also want the secret key as a bearer token
                r = self._http.request(method, self.url + path,
                                       headers=self._headers({**extra, "Authorization": f"Bearer {self._key}"}), **kw)
        except Exception as e:
            raise StoreError(f"Could not reach the project store ({type(e).__name__}). Try again in a minute.")
        if r.status_code >= 400:
            detail = ""
            try:
                body = r.json()
                detail = body.get("message") or body.get("error") or body.get("msg") or ""
            except Exception:
                detail = r.text[:150]
            raise StoreError(f"The project store refused the request ({r.status_code}: {str(detail)[:150]}).")
        return r

    # ------------------------------------------------------------------ projects
    def list(self, owner: str) -> list[dict]:
        self._check_owner(owner)
        r = self._call("GET", f"/rest/v1/{TABLE}", params={"owner": f"eq.{owner}", "select": _COLUMNS,
                                                           "order": "updated_at.desc"})
        return r.json()

    def save(self, owner: str, name: str, df: pd.DataFrame, settings: dict, source: str = "",
             project_id: str | None = None, max_projects: int = MAX_PROJECTS) -> dict:
        self._check_owner(owner)
        name = (name or "").strip()[:80] or "Untitled project"
        buf = io.BytesIO()
        from pipeline import _safe_table
        _safe_table(df).to_parquet(buf, index=False, compression="zstd")
        blob = self._fernet.encrypt(buf.getvalue())
        if len(blob) > MAX_STORED_MB * 1024 * 1024:
            raise StoreError(f"This data is too large to save (over {MAX_STORED_MB} MB after compression).")
        now = _now()
        meta = {"name": name, "source": (source or "")[:200], "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
                "size_bytes": len(blob), "settings": settings,
                "updated_at": _iso(now), "expires_at": _iso(now + dt.timedelta(days=RETENTION_DAYS))}
        if project_id:
            self._check_id(project_id)
            existing = self._get_row(owner, project_id)
            path = existing["storage_path"]
            self._upload(path, blob)
            r = self._call("PATCH", f"/rest/v1/{TABLE}", params={"id": f"eq.{project_id}", "owner": f"eq.{owner}"},
                           json=meta, headers={"Prefer": "return=representation"})
            return r.json()[0]
        if len(self.list(owner)) >= max_projects:
            raise StoreError(f"You have {max_projects} saved projects, the limit for your plan. "
                             "Delete one to save another" + (", or upgrade to Pro." if max_projects < MAX_PROJECTS else "."))
        pid = str(uuid.uuid4())
        path = f"{owner}/{pid}.bin"
        self._upload(path, blob)
        row = {"id": pid, "owner": owner, "storage_path": path, "created_at": _iso(now), **meta}
        try:
            r = self._call("POST", f"/rest/v1/{TABLE}", json=row, headers={"Prefer": "return=representation"})
        except StoreError:
            self._remove([path])
            raise
        return r.json()[0]

    def load(self, owner: str, project_id: str) -> tuple[pd.DataFrame, dict]:
        self._check_owner(owner)
        self._check_id(project_id)
        row = self._get_row(owner, project_id)
        r = self._call("GET", f"/storage/v1/object/{BUCKET}/{row['storage_path']}")
        try:
            raw = self._fernet.decrypt(r.content)
        except Exception:
            raise StoreError("This project could not be decrypted. The encryption key in Secrets may have changed.")
        df = pd.read_parquet(io.BytesIO(raw))
        now = _now()                                     # opening it keeps it for another RETENTION_DAYS
        self._call("PATCH", f"/rest/v1/{TABLE}", params={"id": f"eq.{project_id}", "owner": f"eq.{owner}"},
                   json={"expires_at": _iso(now + dt.timedelta(days=RETENTION_DAYS))})
        return df, row

    def delete(self, owner: str, project_id: str) -> None:
        self._check_owner(owner)
        self._check_id(project_id)
        row = self._get_row(owner, project_id)
        self._remove([row["storage_path"]])
        self._call("DELETE", f"/rest/v1/{TABLE}", params={"id": f"eq.{project_id}", "owner": f"eq.{owner}"})

    def delete_all(self, owner: str) -> int:
        self._check_owner(owner)
        r = self._call("GET", f"/rest/v1/{TABLE}", params={"owner": f"eq.{owner}", "select": "id,storage_path"})
        rows = r.json()
        if rows:
            self._remove([x["storage_path"] for x in rows])
            self._call("DELETE", f"/rest/v1/{TABLE}", params={"owner": f"eq.{owner}"})
        return len(rows)

    def cleanup_expired(self, force: bool = False, batch: int = 50) -> int:
        """Delete projects not opened for RETENTION_DAYS. Runs at most every 6 hours per server."""
        if not force and time.time() - self._last_cleanup < 6 * 3600:
            return 0
        self._last_cleanup = time.time()
        r = self._call("GET", f"/rest/v1/{TABLE}", params={"expires_at": f"lt.{_iso(_now())}",
                                                           "select": "id,storage_path", "limit": str(batch)})
        rows = r.json()
        if rows:
            self._remove([x["storage_path"] for x in rows])
            ids = ",".join(x["id"] for x in rows if _UUID.match(x["id"]))
            self._call("DELETE", f"/rest/v1/{TABLE}", params={"id": f"in.({ids})"})
        return len(rows)

    # ------------------------------------------------------------------ billing link (update 28)
    def get_customer(self, owner: str) -> str | None:
        self._check_owner(owner)
        r = self._call("GET", "/rest/v1/billing", params={"owner": f"eq.{owner}", "select": "stripe_customer_id"})
        rows = r.json()
        return rows[0]["stripe_customer_id"] if rows else None

    def set_customer(self, owner: str, customer_id: str) -> None:
        self._check_owner(owner)
        if not re.match(r"^cus_[A-Za-z0-9]+$", customer_id or ""):
            raise StoreError("Unexpected customer id from the payment provider.")
        self._call("POST", "/rest/v1/billing", json={"owner": owner, "stripe_customer_id": customer_id},
                   headers={"Prefer": "resolution=merge-duplicates"})

    def delete_customer_link(self, owner: str) -> None:
        self._check_owner(owner)
        self._call("DELETE", "/rest/v1/billing", params={"owner": f"eq.{owner}"})

    def check(self) -> list[tuple[str, bool, str]]:
        """Admin diagnostics: (step, ok, note)."""
        out = []
        try:
            self._call("GET", f"/rest/v1/{TABLE}", params={"select": "id", "limit": "1"})
            out.append(("Projects table", True, "reachable"))
        except StoreError as e:
            out.append(("Projects table", False, str(e)))
        try:
            probe = f"_healthcheck/{uuid.uuid4()}.bin"
            self._upload(probe, self._fernet.encrypt(b"ok"))
            ok = self._fernet.decrypt(self._call("GET", f"/storage/v1/object/{BUCKET}/{probe}").content) == b"ok"
            self._remove([probe])
            out.append(("Private file bucket", ok, "write, read and delete work" if ok else "read back failed"))
        except StoreError as e:
            out.append(("Private file bucket", False, str(e)))
        try:
            import httpx
            pub = httpx.get(f"{self.url}/storage/v1/object/public/{BUCKET}/_healthcheck/none.bin", timeout=10)
            out.append(("Bucket is private", pub.status_code >= 400, f"public link returns {pub.status_code}"))
        except Exception as e:
            out.append(("Bucket is private", False, type(e).__name__))
        return out

    # ------------------------------------------------------------------ helpers
    def _get_row(self, owner: str, project_id: str) -> dict:
        r = self._call("GET", f"/rest/v1/{TABLE}", params={"id": f"eq.{project_id}", "owner": f"eq.{owner}",
                                                           "select": _COLUMNS + ",storage_path,settings"})
        rows = r.json()
        if not rows:
            raise StoreError("That project was not found. It may have been deleted.")
        return rows[0]

    def _upload(self, path: str, blob: bytes) -> None:
        self._call("POST", f"/storage/v1/object/{BUCKET}/{path}", content=blob,
                   headers={"Content-Type": "application/octet-stream", "x-upsert": "true"})

    def _remove(self, paths: list[str]) -> None:
        if paths:
            self._call("DELETE", f"/storage/v1/object/{BUCKET}", json={"prefixes": paths})

    @staticmethod
    def _check_owner(owner: str):
        if not _HEX.match(owner or ""):
            raise StoreError("Sign in to use saved projects.")

    @staticmethod
    def _check_id(pid: str):
        if not _UUID.match(pid or ""):
            raise StoreError("That project id is not valid.")


SETUP_SQL = f"""-- Vanguard Analytics, update 27: run once in Supabase -> SQL Editor
create table if not exists public.{TABLE} (
  id           uuid primary key default gen_random_uuid(),
  owner        text not null check (owner ~ '^[0-9a-f]{{64}}$'),
  name         text not null check (char_length(name) <= 80),
  source       text,
  n_rows       integer,
  n_cols       integer,
  size_bytes   bigint,
  storage_path text not null,
  settings     jsonb not null default '{{}}'::jsonb,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  expires_at   timestamptz not null default now() + interval '{RETENTION_DAYS} days'
);
create index if not exists {TABLE}_owner_updated on public.{TABLE} (owner, updated_at desc);
create index if not exists {TABLE}_expires on public.{TABLE} (expires_at);

-- Locked down: row level security on, no policies, nothing granted to the public roles.
alter table public.{TABLE} enable row level security;
revoke all on public.{TABLE} from anon, authenticated;
grant select, insert, update, delete on public.{TABLE} to service_role;

-- Update 28: link between an account code and its Stripe customer (no card data is ever stored here).
create table if not exists public.billing (
  owner              text primary key check (owner ~ '^[0-9a-f]{{64}}$'),
  stripe_customer_id text not null unique check (stripe_customer_id ~ '^cus_[A-Za-z0-9]+$'),
  created_at         timestamptz not null default now()
);
alter table public.billing enable row level security;
revoke all on public.billing from anon, authenticated;
grant select, insert, update, delete on public.billing to service_role;

-- Private bucket for the encrypted project files (50 MB per file).
insert into storage.buckets (id, name, public, file_size_limit)
values ('{BUCKET}', '{BUCKET}', false, {MAX_STORED_MB * 1024 * 1024})
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;
"""
