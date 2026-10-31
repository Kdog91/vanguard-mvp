"""
Sign-in (Streamlit's built-in OpenID Connect login): Google directly, or Auth0, which lets anyone sign up
with email and password or with Google, Microsoft and other accounts.

Anyone can open the app and try the sample data and public USAspending data. Loading your own data
(upload, web address, database, large file) requires signing in. Optionally only approved emails
are let in.

Setup (Streamlit Cloud: app → Settings → Secrets; on your computer: .streamlit/secrets.toml):

    [auth]
    redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
    cookie_secret = "a long random string"
    client_id = "....apps.googleusercontent.com"
    client_secret = "...."
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

    [access]                      # optional
    allowed_emails = ["you@example.com", "client@agency.gov"]
    allowed_domains = ["agency.gov"]
    admin_emails = ["vanguarddataanalytics@gmail.com"]

For Auth0 use server_metadata_url = "https://YOUR-TENANT.us.auth0.com/.well-known/openid-configuration"
and the Auth0 application's client_id and client_secret.

Without an [auth] section the app runs open (for local development) and says so.
"""
import streamlit as st


def _secrets(section: str) -> dict:
    try:
        return dict(st.secrets.get(section, {}))
    except Exception:          # no secrets file at all
        return {}


def configured() -> bool:
    a = _secrets("auth")
    return bool(a.get("client_id") and a.get("client_secret") and a.get("redirect_uri"))


def authlib_ok() -> tuple[bool, str]:
    try:
        import authlib
        return True, getattr(authlib, "__version__", "?")
    except Exception as e:
        return False, str(e)


def logged_in() -> bool:
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def email() -> str:
    try:
        return (st.user.get("email") or "").lower()
    except Exception:
        return ""


def name() -> str:
    try:
        return st.user.get("name") or email()
    except Exception:
        return ""


def subject() -> str:
    """The sign-in provider's permanent id for this account (e.g. google-oauth2|1234 or auth0|abcd).
    Unlike the email, two different accounts can never share it, so saved projects are keyed on it."""
    try:
        return str(st.user.get("sub") or "")
    except Exception:
        return ""


def email_verified() -> bool:
    try:
        v = st.user.get("email_verified")
    except Exception:
        return False
    return v is True or str(v).lower() == "true"


def approved() -> bool:
    """Optional allow-list. admin_emails only marks admins; it does not limit who may sign in."""
    acc = _secrets("access")
    emails = [e.lower() for e in acc.get("allowed_emails", [])]
    domains = [d.lower().lstrip("@") for d in acc.get("allowed_domains", [])]
    if is_admin():
        return True
    if not emails and not domains:
        return True                                   # anyone who signs up
    e = email()
    if not email_verified():
        return False                                  # allow-lists only trust verified emails
    return e in emails or (e.split("@")[-1] in domains if "@" in e else False)


def is_admin() -> bool:
    """Admin = an email in [access] admin_emails that the sign-in provider has verified."""
    admins = [e.lower() for e in _secrets("access").get("admin_emails", [])]
    return bool(admins) and logged_in() and email_verified() and email() in admins


def can_use_own_data() -> bool:
    if not configured():
        return True                                   # local development: open
    return logged_in() and approved()


def user_id() -> str:
    """Stable id for saved projects: the email, or 'local' when sign-in is not set up."""
    return email() if (configured() and logged_in()) else "local"


def render_sidebar():
    """Account box at the top of the sidebar."""
    if not configured():
        st.caption("Sign-in is not set up, so this copy is open. Do not load private data on a shared link.")
        return
    ok, info = authlib_ok()
    if not ok:
        st.error("Sign-in is set up but the Authlib package is missing on this server, so sign-in cannot start. "
                 "Reboot the app so it reinstalls requirements.txt. (" + info[:80] + ")")
        return
    if logged_in():
        if not st.session_state.get("_audit_signin"):
            st.session_state["_audit_signin"] = True
            import security
            security.audit("sign_in", approved=approved())
        st.markdown(f"**Signed in** as {name()}")
        if name() != email():
            st.caption(email())
        if not approved():
            st.warning("This account is not approved yet. Contact vanguarddataanalytics@gmail.com for access. "
                       "You can still try the sample data.")
        st.button("Sign out", key="auth_out", on_click=_sign_out)
    else:
        label = "Sign in with Google" if "google" in _secrets("auth").get("server_metadata_url", "") else "Sign in / Sign up"
        st.button(label, key="auth_in", type="primary", on_click=st.login, use_container_width=True)
        st.caption("Free account. Sign in to load your own data. Sample data works without signing in.")


def _sign_out():
    import security
    security.audit("sign_out")
    st.logout()


def render_locked(what: str = "your own data"):
    st.info(f"Sign in (top of this panel) to load {what}. Your files are only visible to you.")


def render_check():
    """?page=authcheck : sign-in diagnostics without showing any secret values."""
    import importlib, sys
    st.title("Sign-in check")
    a = _secrets("auth")
    rows = []
    for k in ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url"):
        v = str(a.get(k, ""))
        shown = v if k in ("redirect_uri", "server_metadata_url") else (f"set ({len(v)} characters)" if v else "MISSING")
        bad = ""
        if v != v.strip():
            bad = "has spaces at start or end"
        elif any(q in v for q in "\u201c\u201d"):
            bad = "has curly quotes"
        rows.append((k, shown, bad or "ok"))
    extra = [k for k in a if k not in ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")]
    st.table({"Setting": [r[0] for r in rows], "Value": [r[1] for r in rows], "Check": [r[2] for r in rows]})
    if extra:
        st.warning("Unexpected keys in [auth]: " + ", ".join(extra))
    for mod in ("streamlit", "authlib", "httpx"):
        try:
            m = importlib.import_module(mod)
            st.write(f"{mod}: {getattr(m, '__version__', 'installed')}")
        except Exception as e:
            st.error(f"{mod}: NOT INSTALLED ({e})")
    url = a.get("server_metadata_url", "")
    if url:
        try:
            import urllib.request, json
            with urllib.request.urlopen(url, timeout=10) as r:
                meta = json.loads(r.read())
            st.success("Sign-in provider reachable: " + meta.get("issuer", "?"))
        except Exception as e:
            st.error(f"Cannot reach server_metadata_url: {e}")
    try:
        from streamlit.auth_util import validate_auth_credentials  # noqa
    except Exception:
        pass
    st.write("Python", sys.version.split()[0])
