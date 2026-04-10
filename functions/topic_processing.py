"""
topic_processing.py
-------------------
Functions for post-processing raw BERTopic output into cleaned, grouped
topic assignments.

The two key operations are:
1. **Filter bad topics** – remove noise clusters (-1, artefact topics).
2. **Collapse duplicate topics** – merge semantically similar topics into
   a single canonical representative.

Both operations keep the ``topic_number`` and ``probability`` list columns
aligned so downstream analysis remains correct.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from config import BAD_TOPICS, SAME_TOPICS, TOPIC_MAP, TOPIC_GROUP_MAP, TOPIC_COLOURS


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------

def _filter_bad_topics(row: pd.Series, bad: set[int]) -> pd.Series:
    """Remove bad topic ids (and their probabilities) from a single row."""
    filtered = [
        (t, p)
        for t, p in zip(row["topic_number"], row["probability"])
        if t not in bad
    ]
    if filtered:
        row["topic_number"], row["probability"] = map(list, zip(*filtered))
    else:
        row["topic_number"] = []
        row["probability"]  = []
    return row


def _collapse_topics(row: pd.Series) -> pd.Series:
    """Remap each topic id to its canonical id."""
    row["topic_number"] = [TOPIC_MAP.get(t, t) for t in row["topic_number"]]
    return row


def _merge_duplicate_max(row: pd.Series) -> pd.Series:
    """After collapsing, keep only the highest probability per canonical topic."""
    acc: dict[int, float] = defaultdict(lambda: float("-inf"))
    for t, p in zip(row["topic_number"], row["probability"]):
        if isinstance(p, float) and math.isnan(p):
            continue
        acc[t] = max(acc[t], p)
    row["topic_number"] = list(acc.keys())
    row["probability"]  = list(acc.values())
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_topics(
    tp_df: pd.DataFrame,
    bad_topics: list[int] = BAD_TOPICS,
) -> pd.DataFrame:
    """Apply all topic post-processing steps and return a cleaned copy.

    Steps applied in order:
    1. Filter rows to those where ``topic_number`` and ``probability`` are
       list-typed (parses string representations if needed).
    2. Remove bad / noise topics.
    3. Collapse duplicate topics to their canonical id.
    4. Deduplicate within each row, keeping the max probability.

    Parameters
    ----------
    tp_df:
        Raw topic-assignment DataFrame from the database.
    bad_topics:
        List of topic ids to discard entirely.

    Returns
    -------
    pd.DataFrame
        A copy of ``tp_df`` with cleaned ``topic_number`` and ``probability``
        columns.
    """
    import ast

    df = tp_df.copy()

    # Parse string representations if the DB returns them as text
    for col in ("topic_number", "probability"):
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    bad = set(bad_topics)
    df = df.apply(_filter_bad_topics, axis=1, bad=bad)
    df = df.apply(_collapse_topics, axis=1)
    df = df.apply(_merge_duplicate_max, axis=1)
    return df


def build_topic_names(
    raw_topic_names: pd.DataFrame,
    bad_topics: list[int] = BAD_TOPICS,
) -> pd.DataFrame:
    """Return a cleaned topic-names table with group labels and colours.

    Parameters
    ----------
    raw_topic_names:
        DataFrame loaded from ``object_topics.csv``; must have columns
        ``Topic``, ``Count``, ``Name``.
    bad_topics:
        Topic ids to remove.

    Returns
    -------
    pd.DataFrame
        Columns: ``Topic``, ``Count``, ``Name``, ``topic_group``, ``color``.
        Sorted by ``Count`` descending.
    """
    df = raw_topic_names[~raw_topic_names["Topic"].isin(bad_topics)].copy()

    # Remap each topic id to its canonical representative
    df["Topic"] = df["Topic"].map(lambda t: TOPIC_MAP.get(t, t))

    # Aggregate counts for merged topics
    df = (
        df.groupby("Topic", as_index=False)
        .agg(Count=("Count", "sum"), Name=("Name", "first"))
    )

    # Attach group names and colours
    df["topic_group"] = df["Topic"].map(TOPIC_GROUP_MAP)
    df["color"] = [
        TOPIC_COLOURS[i % len(TOPIC_COLOURS)] for i in range(len(df))
    ]
    df = df.sort_values("Count", ascending=False).reset_index(drop=True)
    return df
