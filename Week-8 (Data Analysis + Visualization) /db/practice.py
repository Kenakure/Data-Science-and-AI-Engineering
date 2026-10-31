from __future__ import annotations

import copy
import datetime
import decimal
import hashlib
import json
import math
import re
from typing import Any, Callable, Sequence

import pandas as pd

from db.dataraflow_db import run_query

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print


_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE|DENY|INTO|BACKUP|RESTORE)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")


def _strip_sql(sql: str) -> str:
    """Remove string literals and comments, so the safety check only sees real keywords."""
    sql = re.sub(r"'([^']|'')*'", "''", sql)
    sql = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.S)


def q(sql: str) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame."""
    hit = _BLOCKED.search(_strip_sql(sql))
    if hit:
        raise PermissionError(
            f"'{hit.group(0).upper()}' is not allowed: the database is read-only."
        )
    return run_query(sql)


def _norm(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, (datetime.date, datetime.datetime, pd.Timestamp)):
        return pd.Timestamp(v).date().isoformat()
    if isinstance(v, str):
        v = v.strip()
        if _DATE.match(v):
            try:
                return pd.Timestamp(v).date().isoformat()
            except ValueError:
                pass
        return v
    if isinstance(v, (bool, int, float, decimal.Decimal)) or hasattr(v, "float"):
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            pass
    return str(v)


def fingerprint(df: pd.DataFrame) -> str:
    """Hash of a result's values (row order and column order matter; names don't)."""
    rows = [[_norm(v) for v in row] for row in df.itertuples(index=False)]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _unordered(got, expected) -> bool:
    try:
        return sorted(map(repr, got)) == sorted(map(repr, expected))
    except TypeError:
        return False


COMPARERS: dict[str, Callable[[Any, Any], bool]] = {
    "exact": lambda got, exp: got == exp,
    "unordered": _unordered,  # same items, any order
    "float": lambda got, exp: got is not None
    and abs(float(got) - float(exp)) <= 1e-6,
}

class Practice: 
    """Tracks SQL and DSA results for one Take Home."""

    def __init__(self, expected_sql: dict | None = None, titles: dict | None = None):
        self.expected = expected_sql or {}
        self.titles = titles or {}
        self.results: dict[str, bool | None] = {}

    # ---------------------------------------------------------------- SQL
    def sql(self, task: str, query: str, show: int = 15):
        """Run your query for an SQL task, show the result and check it."""
        shape, fp = self.expected[task]
        if not _strip_sql(query).strip():
            self.results[task] = None
            print(f"⏳ {task}: not attempted yet. Write your query between the triple quotes.")
            return None
        try:
            df = q(query)
        except Exception as err:  # show the database's message instead of a long traceback
            self.results[task] = False
            print(f"❌ {task}: the query failed.\n   {type(err).__name__}: {str(err)[:300]}")
            return None
        display(df.head(show))
        ok = fingerprint(df) == fp
        self.results[task] = ok
        if ok:
            print(f"✅ {task}: correct!")
        else:
            print(f"❌ {task}: not quite. Expected {shape[0]} row(s) × {shape[1]} column(s); "
                f"you returned {df.shape[0]} × {df.shape[1]}.")
            if tuple(df.shape) == tuple(shape):
                print("   The shape is right, so check: column order, sort order, rounding, and filters "
                    "(NULLs, aggregate rows, duplicates).")
        return df

    # ---------------------------------------------------------------- DSA
    def dsa(self, task: str, fn: Callable, cases: Sequence[tuple], compare: str = "exact"):
        """Run your function against test cases: [((args...), expected), ...]."""
        cmp = COMPARERS[compare]
        passed, lines = 0, []
        for args, expected in cases:
            call_args = args if isinstance(args, tuple) else (args,)
            try:
                got = fn(*copy.deepcopy(call_args))
                ok = cmp(got, expected)
            except NotImplementedError:
                self.results[task] = None
                print(f"⏳ {task}: not attempted yet. Replace `raise NotImplementedError` with your code.")
                return None
            except Exception as err:
                got, ok = f"{type(err).__name__}: {err}", False
            if ok:
                passed += 1
            else:
                lines.append(f"   ❌ {fn.__name__}{call_args!r}\n        expected {expected!r}\n        got      {got!r}")
        ok = passed == len(cases)
        self.results[task] = ok
        print(f"{'✅' if ok else '❌'} {task}: {passed}/{len(cases)} tests passed" + ("!" if ok else ""))
        if lines:
            print("\n".join(lines[:5]))
        return ok

    # ---------------------------------------------------------------- summary
    def score(self):
        """Print a table of every SQL and DSA task and your score."""
        rows = []
        for task, title in self.titles.items():
            r = self.results.get(task)
            kind = "SQL" if task.startswith("s") else "DSA"
            rows.append([task, kind, title, "✅" if r else ("not attempted" if r is None else "❌")])
        table = pd.DataFrame(rows, columns=["Task", "Type", "Topic", "Result"])
        print(table.to_string(index=False))
        for kind in ("SQL", "DSA"):
            part = table[table["Type"] == kind]
            if len(part):
                print(f"{kind} score: {(part['Result'] == '✅').sum()} / {len(part)}")
        return table