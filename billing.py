"""
Plans and payments (update 28): Free and Pro ($29/month) with Stripe.

- Payment happens on Stripe Checkout (Stripe's own page), so card numbers never reach this app (PCI DSS SAQ A).
- Customers manage or cancel in Stripe's customer portal.
- No webhooks: the app asks Stripe directly whether an account has an active subscription (cached for a few
  minutes), so there is no public endpoint to forge and nothing to keep in sync.
- The only billing data stored by the app is the link between an account code and its Stripe customer id.

Secrets:
    [stripe]
    secret_key = "rk_test_..."        # restricted key (preferred) or sk_test_...; live keys start rk_live_/sk_live_
    price_id = "price_..."            # the $29/month Pro price
    app_url = "https://YOUR-APP.streamlit.app"
"""
from __future__ import annotations

import re
import time

import streamlit as st

import auth

PLANS = {
    "free": {"name": "Free", "price": "$0", "max_upload_mb": 5, "max_rows": 50_000, "max_projects": 2,
             "deep_learning": False},
    "pro": {"name": "Pro", "price": "$29/month", "max_upload_mb": 200, "max_rows": 1_000_000, "max_projects": 25,
            "deep_learning": True},
}
PRO_FEATURES = ["Files up to 200 MB and 1,000,000 rows", "25 saved projects",
                "Deep learning: neural networks, image models, model export with FastAPI + Docker",
                "Everything in Free"]
FREE_FEATURES = ["Files up to 5 MB and 50,000 rows", "2 saved projects",
                 "Cleaning, statistics, prediction, forecasting, reports, pipelines"]
ACTIVE = ("active", "trialing", "past_due")          # past_due: Stripe is retrying the card; keep access meanwhile
CACHE_SECONDS = 300
API = "https://api.stripe.com/v1"


class BillingError(RuntimeError):
    pass


def _secrets() -> dict:
    try:
        return dict(st.secrets.get("stripe", {}))
    except Exception:
        return {}


def configured() -> bool:
    """Stripe keys present, and saved projects (Supabase) set up, since that holds the customer link."""
    import projects_ui
    s = _secrets()
    return bool(s.get("secret_key") and s.get("price_id") and s.get("app_url")) and projects_ui.configured()


def test_mode() -> bool:
    return "_test_" in str(_secrets().get("secret_key", ""))


# ------------------------------------------------------------------ Stripe REST
def _stripe(method: str, path: str, data: dict | None = None, params: dict | None = None) -> dict:
    import httpx
    key = _secrets().get("secret_key", "")
    if not re.match(r"^(sk|rk)_(test|live)_[A-Za-z0-9]+$", key):
        raise BillingError("The Stripe key in Secrets is not a secret or restricted key (sk_... or rk_...).")
    try:
        r = httpx.request(method, API + path, data=data, params=params, auth=(key, ""), timeout=20,
                          headers={"Stripe-Version": "2024-06-20"})
    except Exception as e:
        raise BillingError(f"Could not reach Stripe ({type(e).__name__}). Try again in a minute.")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 400:
        msg = (body.get("error") or {}).get("message", "") if isinstance(body, dict) else ""
        raise BillingError(f"Stripe refused the request ({r.status_code}). {msg[:160]}")
    return body


def _owner():
    import projects_ui
    return projects_ui._owner()


def _store():
    import projects_ui
    return projects_ui.get_store()


def _customer(create: bool = False) -> str | None:
    owner, s = _owner(), _store()
    if owner is None or s is None:
        return None
    cid = s.get_customer(owner)
    if cid or not create:
        return cid
    c = _stripe("POST", "/customers", data={"email": auth.email() or "", "name": auth.name() or "",
                                           "metadata[owner]": owner, "metadata[app]": "vanguard-analytics"})
    s.set_customer(owner, c["id"])
    return c["id"]


def _lookup_status(cid: str) -> str:
    subs = _stripe("GET", "/subscriptions", params={"customer": cid, "status": "all", "limit": "10"})
    price = _secrets().get("price_id")
    statuses = [x["status"] for x in subs.get("data", [])
                if any((it.get("price") or {}).get("id") == price for it in (x.get("items") or {}).get("data", []))]
    return next((st_ for st_ in statuses if st_ in ACTIVE), statuses[0] if statuses else "none")


# ------------------------------------------------------------------ the plan in force
def plan_key() -> str:
    """'pro' or 'free'. Without Stripe set up (or for admins and local copies) everything is unlocked."""
    if not configured() or not auth.configured() or auth.is_admin():
        return "pro"
    if not auth.logged_in():
        return "free"
    cache = st.session_state.get("_plan_cache")
    if cache and time.time() - cache[1] < CACHE_SECONDS:
        return cache[0]
    try:
        cid = _customer()
        status = _lookup_status(cid) if cid else "none"
        key = "pro" if status in ACTIVE else "free"
        st.session_state["_plan_status"] = status
    except Exception as e:
        key = (cache or ("free",))[0]                    # Stripe unreachable: keep the last known plan
        st.session_state["_plan_error"] = str(e)[:200]
    st.session_state["_plan_cache"] = (key, time.time())
    return key


def limits() -> dict:
    return PLANS[plan_key()]


def refresh():
    st.session_state.pop("_plan_cache", None)


# ------------------------------------------------------------------ checkout and portal
def checkout_url() -> str:
    s = _secrets()
    owner = _owner()
    if owner is None:
        raise BillingError("Sign in first.")
    cid = _customer(create=True)
    base = s["app_url"].rstrip("/")
    sess = _stripe("POST", "/checkout/sessions", data={
        "mode": "subscription", "customer": cid, "client_reference_id": owner,
        "line_items[0][price]": s["price_id"], "line_items[0][quantity]": "1",
        "allow_promotion_codes": "true", "billing_address_collection": "auto",
        "success_url": base + "/?checkout=success&session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": base + "/?checkout=cancel",
        "subscription_data[metadata][owner]": owner,
    })
    return sess["url"]


def portal_url() -> str:
    cid = _customer()
    if not cid:
        raise BillingError("No billing account yet.")
    return _stripe("POST", "/billing_portal/sessions",
                   data={"customer": cid, "return_url": _secrets()["app_url"].rstrip("/") + "/"})["url"]


def handle_return():
    """After Stripe sends the customer back: confirm the checkout belongs to this account, then refresh the plan."""
    qp = st.query_params
    state = qp.get("checkout")
    if not state:
        return
    import security
    if state == "success":
        sid = qp.get("session_id", "")
        try:
            if not re.match(r"^cs_(test|live)_[A-Za-z0-9]+$", sid):
                raise BillingError("Unexpected checkout reference.")
            sess = _stripe("GET", f"/checkout/sessions/{sid}")
            if sess.get("client_reference_id") != _owner():
                raise BillingError("That checkout belongs to a different account.")
            refresh()
            if sess.get("status") == "complete":
                st.session_state["billing_msg"] = ("success", "Welcome to Pro! Your plan is active.")
                security.audit("checkout_success")
            else:
                st.session_state["billing_msg"] = ("info", "Payment is still processing. Your plan will update shortly.")
        except BillingError as e:
            st.session_state["billing_msg"] = ("error", str(e))
    elif state == "cancel":
        st.session_state["billing_msg"] = ("info", "Checkout was cancelled. You were not charged.")
    st.query_params.clear()


# ------------------------------------------------------------------ UI
def render_sidebar():
    if not (configured() and auth.configured()):
        return
    st.markdown("### Plan")
    msg = st.session_state.pop("billing_msg", None)
    if msg:
        {"success": st.success, "error": st.error}.get(msg[0], st.info)(msg[1])
    if test_mode():
        st.caption(":orange[TEST MODE: no real charges. Use card 4242 4242 4242 4242.]")
    if not auth.logged_in():
        st.caption("Free to start. Pro is $29/month.")
        return
    key = plan_key()
    p = PLANS[key]
    if auth.is_admin():
        st.caption("**Pro** (administrator)")
        return
    st.caption(f"**{p['name']}** · {p['price']}"
               + (f" · status: {st.session_state.get('_plan_status')}" if key == "pro" else ""))
    if key == "pro":
        if st.button("Manage billing", key="bill_portal", use_container_width=True):
            _go(portal_url, "portal_open")
    else:
        with st.expander("What Pro adds"):
            for f in PRO_FEATURES:
                st.markdown(f"- {f}")
        if st.button("Upgrade to Pro · $29/month", key="bill_up", type="primary", use_container_width=True):
            _go(checkout_url, "checkout_start")
        if st.session_state.get("_plan_status") in ("canceled", "unpaid", "incomplete_expired") and _customer():
            if st.button("Billing history", key="bill_hist", use_container_width=True):
                _go(portal_url, "portal_open")


def _go(make_url, event: str):
    import security
    try:
        url = make_url()
    except BillingError as e:
        st.error(str(e))
        return
    security.audit(event)
    st.link_button("Continue to Stripe →", url, type="primary", use_container_width=True)
    st.caption("Opens Stripe's secure page. Card details go to Stripe, never to this app.")


def upgrade_box(reason: str):
    """Shown where a Free limit is reached."""
    st.warning(f"{reason} This is part of **Pro ($29/month)**.")
    if configured() and auth.logged_in():
        if st.button("Upgrade to Pro", key=f"up_{abs(hash(reason)) % 10**6}", type="primary"):
            _go(checkout_url, "checkout_start")
    else:
        st.caption("Sign in, then choose **Upgrade to Pro** in the sidebar.")
