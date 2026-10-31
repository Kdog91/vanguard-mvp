"""Saved projects in the app: 'My projects' in the sidebar, 'Save this project' under the data, admin check."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import auth
import security
import store


def _secrets() -> dict:
    try:
        return dict(st.secrets.get("supabase", {}))
    except Exception:
        return {}


def configured() -> bool:
    s = _secrets()
    return bool(s.get("url") and s.get("secret_key") and s.get("encryption_key"))


@st.cache_resource(show_spinner=False)
def _store(url: str, key: str, enc: str):
    return store.SupabaseStore(url, key, enc)


def get_store():
    if not configured():
        return None
    s = _secrets()
    return _store(s["url"], s["secret_key"], s["encryption_key"])


def _owner() -> str | None:
    if auth.configured():
        if not (auth.logged_in() and auth.approved() and auth.subject()):
            return None
        return store.owner_code(auth.subject())
    return store.owner_code("local")             # your own computer without sign-in


def _when(ts: str) -> str:
    try:
        return pd.Timestamp(ts).tz_convert("America/Chicago").strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")
    except Exception:
        return str(ts)[:16]


# ------------------------------------------------------------------ open / delete (button callbacks)
def _open(pid: str):
    try:
        df, row = get_store().load(_owner(), pid)
    except store.StoreError as e:
        st.session_state["proj_msg"] = ("error", str(e))
        return
    s = row.get("settings") or {}
    label = s.get("label") or "project"
    for k in ("usa_data",):
        st.session_state.pop(k, None)
    st.session_state["sample_choice"] = "None"
    st.session_state["up_ver"] = st.session_state.get("up_ver", 0) + 1          # clears the upload box
    st.session_state["url_data"] = {"df": df, "url": f"Saved project: {row['name']}", "kind": "project",
                                    "label": label, "project_id": pid, "project_name": row["name"]}
    st.session_state[f"overrides::{label}"] = s.get("overrides") or {}
    if s.get("missing_choice"):
        st.session_state[f"miss_{label}"] = s["missing_choice"]
    st.session_state[f"hdr_{label}"] = False
    st.session_state[f"names_{label}"] = ""
    if s.get("mode"):
        st.session_state["mode"] = s["mode"]
        if s.get("advanced"):
            st.session_state["show_advanced"] = True
    st.session_state["proj_msg"] = ("success", f"Opened '{row['name']}'.")
    security.audit("project_open", rows=len(df))


def _delete(pid: str, name: str):
    try:
        get_store().delete(_owner(), pid)
    except store.StoreError as e:
        st.session_state["proj_msg"] = ("error", str(e))
        return
    cur = st.session_state.get("url_data") or {}
    if cur.get("project_id") == pid:
        st.session_state.pop("url_data", None)
    st.session_state["proj_msg"] = ("success", f"Deleted '{name}'.")
    security.audit("project_delete")


def _delete_all():
    try:
        n = get_store().delete_all(_owner())
    except store.StoreError as e:
        st.session_state["proj_msg"] = ("error", str(e))
        return
    if (st.session_state.get("url_data") or {}).get("kind") == "project":
        st.session_state.pop("url_data", None)
    st.session_state["proj_msg"] = ("success", f"Deleted all {n} saved projects.")
    st.session_state["proj_confirm_all"] = False
    security.audit("project_delete_all", count=n)


# ------------------------------------------------------------------ sidebar
def render_sidebar():
    if not configured():
        return
    st.markdown("### My projects")
    owner = _owner()
    if owner is None:
        st.caption("Sign in to save your work and reopen it later.")
        return
    msg = st.session_state.pop("proj_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
    try:
        s = get_store()
        s.cleanup_expired()
        rows = s.list(owner)
    except store.StoreError as e:
        st.error(str(e))
        return
    if not rows:
        st.caption("No saved projects yet. Load data, then use **Save this project** under it.")
        return
    import billing
    st.caption(f"{len(rows)} of {billing.limits()['max_projects']} · encrypted · deleted after "
               f"{store.RETENTION_DAYS} days unopened")
    for r in rows:
        with st.container(border=True):
            st.markdown(f"**{r['name']}**")
            st.caption(f"{r.get('n_rows') or 0:,} rows × {r.get('n_cols') or 0} columns · saved {_when(r['updated_at'])}")
            c1, c2 = st.columns(2)
            c1.button("Open", key=f"po_{r['id']}", on_click=_open, args=(r["id"],), use_container_width=True)
            c2.button("Delete", key=f"pd_{r['id']}", on_click=_delete, args=(r["id"], r["name"]),
                      use_container_width=True)
    if st.session_state.get("proj_confirm_all"):
        st.warning("Delete every saved project? This cannot be undone.")
        a, b = st.columns(2)
        a.button("Yes, delete all", key="pda_yes", on_click=_delete_all, type="primary", use_container_width=True)
        b.button("Cancel", key="pda_no", on_click=lambda: st.session_state.update(proj_confirm_all=False),
                 use_container_width=True)
    else:
        st.button("Delete all my projects", key="pda", use_container_width=True,
                  on_click=lambda: st.session_state.update(proj_confirm_all=True))


def render_privacy_tools():
    """Sidebar: delete everything the app stores about this account."""
    if not configured():
        return
    owner = _owner()
    if owner is None or not auth.configured():
        return
    with st.expander("Delete my data"):
        import billing
        if billing.configured() and billing.plan_key() == "pro" and not auth.is_admin():
            st.caption("Cancel Pro first under **Plan → Manage billing**, then come back here.")
            return
        st.caption("Deletes all your saved projects and the link to your billing record. Stripe keeps its own "
                   "payment records as tax law requires. To also remove your sign-in account, email "
                   "vanguarddataanalytics@gmail.com from the address you sign in with.")
        sure = st.checkbox("I understand this cannot be undone", key="del_me_ok")
        if st.button("Delete my data", key="del_me", disabled=not sure, type="primary"):
            try:
                n = get_store().delete_all(owner)
                get_store().delete_customer_link(owner)
            except store.StoreError as e:
                st.error(str(e))
                return
            st.session_state.pop("url_data", None)
            billing.refresh()
            security.audit("account_data_deleted", projects=n)
            st.success(f"Deleted {n} saved projects and your billing link.")


# ------------------------------------------------------------------ save
def render_save(raw_df: pd.DataFrame, settings: dict, source: str, default_name: str, is_sample_of_big: bool):
    if not configured():
        return
    owner = _owner()
    with st.expander("Save this project", expanded=False):
        if owner is None:
            st.caption("Sign in to save this data and your cleaning choices, and reopen them later.")
            return
        if is_sample_of_big:
            st.caption("Large-file mode works on a sample of a file on your computer, so it can't be saved here.")
            return
        cur = st.session_state.get("url_data") or {}
        is_open = cur.get("kind") == "project"
        name = st.text_input("Project name", value=cur.get("project_name") or default_name[:80], max_chars=80,
                             key="proj_name")
        import pipeline as _pl
        pii = {c: k for c, k in _pl.detect_pii(raw_df).items() if c in raw_df.columns}
        mask = False
        if pii:
            mask = st.checkbox("Replace private details with codes before saving (recommended)", value=True,
                               key="proj_mask",
                               help="Found: " + ", ".join(f"{c} ({k})" for c, k in pii.items())
                                    + ". The same value always gets the same code, so counts and matches still work.")
        st.caption(f"Saves the loaded data ({len(raw_df):,} rows), your column type and missing-value choices and "
                   f"the analysis you picked. Stored encrypted; only your account can open it; deleted after "
                   f"{store.RETENTION_DAYS} days if not opened. Do not save health records (PHI) or controlled "
                   "government data (CUI).")
        c1, c2 = st.columns(2)
        do_new = c1.button("Save as new project", key="proj_save_new", type="primary", use_container_width=True)
        do_upd = c2.button("Update this project", key="proj_save_upd", disabled=not is_open, use_container_width=True,
                           help="Overwrite the saved project you opened." if is_open else "Open a saved project first.")
        if do_new or do_upd:
            data = _pl.mask_pii(raw_df, pii, "hash", salt="vanguard-project") if mask else raw_df
            s = dict(settings, masked_columns=list(pii) if mask else [], app_update="27")
            try:
                with st.spinner("Encrypting and saving..."):
                    import billing
                    row = get_store().save(owner, name, data, s, source=source,
                                           project_id=cur.get("project_id") if do_upd else None,
                                           max_projects=billing.limits()["max_projects"])
            except store.StoreError as e:
                st.error(str(e))
                security.audit("project_save_failed", reason=str(e)[:80])
                return
            st.success(f"Saved '{row['name']}'. Find it under **My projects** in the sidebar.")
            security.audit("project_save", rows=len(data), masked=mask, update=bool(do_upd))
            if do_new and is_open:
                cur.update(project_id=row["id"], project_name=row["name"])


# ------------------------------------------------------------------ admin page (?page=authcheck)
def render_admin_check():
    st.subheader("Saved projects (Supabase)")
    s = _secrets()
    st.table({"Setting": ["url", "secret_key", "encryption_key"],
              "Value": [s.get("url", "MISSING"),
                        (f"set ({len(s['secret_key'])} characters, "
                         f"{'secret' if str(s['secret_key']).startswith('sb_secret_') else 'legacy JWT' if str(s['secret_key']).startswith('eyJ') else 'unknown type'})")
                        if s.get("secret_key") else "MISSING",
                        f"set ({len(s['encryption_key'])} characters)" if s.get("encryption_key") else "MISSING"]})
    if not configured():
        st.info("Add a [supabase] section to Secrets to turn on saved projects.")
        return
    try:
        results = get_store().check()
    except store.StoreError as e:
        st.error(str(e))
        return
    for step, ok, note in results:
        (st.success if ok else st.error)(f"{step}: {note}")
