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


def approved() -> bool:
    acc = _secrets("access")
    emails = [e.lower() for e in acc.get("allowed_emails", [])] + [e.lower() for e in acc.get("admin_emails", [])]
    domains = [d.lower().lstrip("@") for d in acc.get("allowed_domains", [])]
    if not emails and not domains:
        return True                                   # any signed-in Google account
    e = email()
    return e in emails or (e.split("@")[-1] in domains if "@" in e else False)


def is_admin() -> bool:
    return email() in [e.lower() for e in _secrets("access").get("admin_emails", [])]


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
        st.markdown(f"**Signed in** as {name()}")
        if name() != email():
            st.caption(email())
        if not approved():
            st.warning("This account is not approved yet. Contact vanguarddataanalytics@gmail.com for access. "
                       "You can still try the sample data.")
        st.button("Sign out", key="auth_out", on_click=st.logout)
    else:
        label = "Sign in with Google" if "google" in _secrets("auth").get("server_metadata_url", "") else "Sign in / Sign up"
        st.button(label, key="auth_in", type="primary", on_click=st.login, use_container_width=True)
        st.caption("Free account. Sign in to load your own data. Sample data works without signing in.")


def render_locked(what: str = "your own data"):
    st.info(f"Sign in (top of this panel) to load {what}. Your files are only visible to you.")
