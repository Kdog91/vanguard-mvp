"""
Read from and write to relational databases (PostgreSQL, MySQL, SQLite) with SQLAlchemy,
the pandas equivalent of Spark's spark.read.jdbc / df.write.jdbc.
Credentials are used only for the connection and are never written to files or logs.
"""
import re

import pandas as pd

KINDS = {"PostgreSQL": ("postgresql+psycopg2", 5432), "MySQL": ("mysql+pymysql", 3306), "SQLite (file)": ("sqlite", None)}


def make_url(kind: str, host: str = "", port: int | None = None, database: str = "", user: str = "", password: str = ""):
    from sqlalchemy.engine import URL
    driver, default_port = KINDS[kind]
    if driver == "sqlite":
        return URL.create("sqlite", database=database)
    return URL.create(driver, username=user or None, password=password or None, host=host or "localhost",
                      port=int(port or default_port), database=database or None)


def engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10}
                         if url.drivername.startswith("postgresql") else
                         ({"connect_timeout": 10} if url.drivername.startswith("mysql") else {}))


def describe(url) -> str:
    """Connection label without the password, e.g. postgresql://analyst@db.example.org:5432/sales."""
    if url.drivername == "sqlite":
        return f"sqlite:///{url.database}"
    kind = url.drivername.split("+")[0]
    return f"{kind}://{url.username + '@' if url.username else ''}{url.host}:{url.port}/{url.database or ''}"


def list_tables(url) -> list:
    from sqlalchemy import inspect
    eng = engine(url)
    try:
        ins = inspect(eng)
        out = []
        for schema in ([None] if url.drivername == "sqlite" else
                       [s for s in ins.get_schema_names() if s not in ("information_schema", "pg_catalog",
                                                                        "mysql", "performance_schema", "sys")]):
            for t in ins.get_table_names(schema=schema):
                out.append(t if schema in (None, "public", url.database) else f"{schema}.{t}")
        return sorted(out)
    finally:
        eng.dispose()


_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def read(url, table: str = "", sql: str = "", limit: int | None = None) -> pd.DataFrame:
    """Read a whole table or the result of one SELECT query."""
    from sqlalchemy import text
    if sql.strip():
        q = sql.strip().rstrip(";")
        if not re.match(r"^(select|with)\b", q, re.I) or ";" in q:
            raise ValueError("Only one SELECT (or WITH ... SELECT) query is allowed.")
    else:
        if not _SAFE_NAME.match(table or ""):
            raise ValueError("Pick a table name (letters, numbers and underscores).")
        q = f"SELECT * FROM {table}"
    if limit:
        q = f"SELECT * FROM ({q}) AS sub LIMIT {int(limit)}"
    eng = engine(url)
    try:
        with eng.connect() as con:
            df = pd.read_sql(text(q), con)
    finally:
        eng.dispose()
    if df.empty:
        raise ValueError("The query returned no rows.")
    return df


def write(df: pd.DataFrame, url, table: str, mode: str = "fail") -> int:
    """mode: fail (stop if the table exists), replace or append, like Spark's write modes error / overwrite / append."""
    if not _SAFE_NAME.match(table or "") or "." in table:
        raise ValueError("Table names may use letters, numbers and underscores, and must start with a letter.")
    from pipeline import _safe_table
    eng = engine(url)
    try:
        _safe_table(df).to_sql(table, eng, if_exists=mode, index=False, chunksize=5000, method="multi")
    finally:
        eng.dispose()
    return len(df)
