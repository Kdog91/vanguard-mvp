"""Streamlit pieces for database connections: a connection form, loading a table, writing results back."""
import streamlit as st

import dbio
import security


def connection_form(prefix: str):
    """Returns a SQLAlchemy URL, or None. The password stays in this browser session only."""
    kind = st.selectbox("Database", list(dbio.KINDS), key=f"{prefix}_kind")
    if kind.startswith("SQLite"):
        path = st.text_input("Path to the .db / .sqlite file", key=f"{prefix}_path",
                             placeholder=r"C:\data\mydata.db",
                             help="Works only when the app runs on your own computer.")
        if security.ON_CLOUD:
            st.caption("SQLite files can't be opened on the public app. Upload the table as a file instead.")
        return dbio.make_url(kind, database=path.strip().strip('"')) if path.strip() else None
    c1, c2 = st.columns([3, 1])
    host = c1.text_input("Host", key=f"{prefix}_host", placeholder="localhost or db.example.org")
    port = c2.number_input("Port", 1, 65535, dbio.KINDS[kind][1], key=f"{prefix}_port")
    database = st.text_input("Database name", key=f"{prefix}_db")
    c3, c4 = st.columns(2)
    user = c3.text_input("User", key=f"{prefix}_user")
    password = c4.text_input("Password", type="password", key=f"{prefix}_pw")
    if not (host and database):
        return None
    return dbio.make_url(kind, host, port, database, user, password)


def render_sidebar_loader():
    with st.expander("Database (PostgreSQL, MySQL, SQLite)"):
        st.caption("Like Spark's read.jdbc: load a table or a SELECT query. Use an account that can only read. "
                   "On the public demo, the database must be reachable from the internet.")
        url = connection_form("dbin")
        if url is None:
            return
        st.session_state["db_url"] = url
        if st.button("List tables", key="dbin_list"):
            try:
                security.check_database(url)
                st.session_state["db_tables"] = dbio.list_tables(url)
            except Exception as e:
                st.error(f"Could not connect: {_clean(e)}")
        tables = st.session_state.get("db_tables") or []
        table = st.selectbox("Table", tables, key="dbin_table") if tables else st.text_input("Table", key="dbin_table_txt")
        sql = st.text_area("Or a SELECT query (optional)", key="dbin_sql", height=70,
                           placeholder="SELECT * FROM sales WHERE year = 2026")
        limit = st.select_slider("Row limit", [1_000, 10_000, 100_000, 1_000_000], value=100_000, key="dbin_limit",
                                 format_func=lambda v: f"{v:,}")
        if st.button("Load from database", key="dbin_load"):
            try:
                security.check_database(url)
                with st.spinner("Reading..."):
                    df = dbio.read(url, table or "", sql, limit)
                security.check_frame(df)
                name = (table or "query").replace(".", "_")
                st.session_state["url_data"] = {"df": df, "url": f"{dbio.describe(url)}/{name}",
                                                "kind": "database"}
                st.success(f"Loaded {len(df):,} rows.")
                security.audit("db_load", kind=url.drivername.split("+")[0], rows=len(df))
            except Exception as e:
                st.error(f"Could not load: {_clean(e)}")
                security.audit("db_load_failed", kind=url.drivername.split("+")[0], reason=type(e).__name__)


def render_writer(tables: dict, label: str):
    """Write bronze / silver / gold back to a database (Spark's write.jdbc)."""
    st.markdown("**Write a table to a database** (like Spark's `df.write.jdbc`)")
    url = connection_form("dbout")
    if url is None:
        if st.session_state.get("db_url") is not None:
            st.caption("Fill in the connection above (it can be the same database you loaded from).")
        return
    names = [k for k, v in tables.items() if v is not None]
    c1, c2, c3 = st.columns(3)
    which = c1.selectbox("Table to write", names, index=names.index("silver") if "silver" in names else 0, key="dbout_which")
    target = c2.text_input("Name in the database", value=f"{label}_{which}"[:60].lower().replace("-", "_"), key="dbout_name")
    mode = c3.selectbox("If it already exists", ["fail", "replace", "append"], key="dbout_mode",
                        format_func=lambda m: {"fail": "stop (error)", "replace": "overwrite", "append": "append"}[m])
    if st.button("Write to database", key="dbout_go"):
        try:
            security.check_database(url)
            with st.spinner("Writing..."):
                n = dbio.write(tables[which], url, target, mode)
            st.success(f"Wrote {n:,} rows to {target} in {dbio.describe(url)}.")
            security.audit("db_write", kind=url.drivername.split("+")[0], rows=n, mode=mode)
        except Exception as e:
            st.error(f"Could not write: {_clean(e)}")


def _clean(e: Exception) -> str:
    """Error text without connection secrets."""
    msg = str(e).split("\n")[0]
    return msg.replace(st.session_state.get("dbin_pw", "") or "\0", "***").replace(
        st.session_state.get("dbout_pw", "") or "\0", "***")[:300]
