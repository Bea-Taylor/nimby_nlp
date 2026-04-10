"""
data_loader.py
--------------
Unified data-loading helpers for the planning comments project.

All notebooks import from here so that connection logic and query
parameters live in exactly one place.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd

from config import (
    APPLICATION_CSV,
    ES_QUERY_PARAMS,
    LONDON_LAD_FILE,
    LONDON_LPA_FILE,
    LSOA_BOUNDARIES,
    CENSUS_AGE_FILE,
    CENSUS_OCCUPATION_FILE,
    CENSUS_TENURE_FILE,
    INNER_LONDON_BOROUGHS,
)

# These imports assume the project's database package is on sys.path
from database.comments import Comments
from database.topics   import Topics
from elastic_search_fncs import ElasticSearchFncs
import preprocessing_fncs as ppf


# ---------------------------------------------------------------------------
# Raw data sources
# ---------------------------------------------------------------------------

def load_comments(env: str = "dev") -> pd.DataFrame:
    """Return the full comments table from the remote database.

    Borough names are normalised via ``ppf.format_borough`` so the
    'council' column is consistent across all notebooks.
    """
    cs = Comments(env=env)
    df = cs.read_all().copy()
    df = ppf.format_borough(df, "council")
    return df


def load_topics(env: str = "dev") -> pd.DataFrame:
    """Return the full topic-assignment table from the remote database."""
    tp = Topics(env=env)
    return tp.read_all()


def load_planning_applications() -> pd.DataFrame:
    """Load the planning-application CSV and drop Bromley (no scraped comments)."""
    df = pd.read_csv(APPLICATION_CSV)
    df = df[df["borough"] != "Bromley"].copy()
    return df


def load_elasticsearch_planning(
    min_res_units: int   = ES_QUERY_PARAMS["min_res_units"],
    since_year:    str   = ES_QUERY_PARAMS["since_year"],
    to_year:       str   = ES_QUERY_PARAMS["to_year"],
) -> pd.DataFrame:
    """Query Elasticsearch for residential planning applications.

    Returns a formatted DataFrame with a 'housing_type' column added.
    """
    esf = ElasticSearchFncs()
    esf.check_connection()
    df = esf.res_units_x_query(
        min_res_units=min_res_units,
        since_year=since_year,
        to_year=to_year,
    )
    df = ppf.format_df(df)
    df = ppf.add_housing_type(df)
    return df


# ---------------------------------------------------------------------------
# Geospatial boundaries
# ---------------------------------------------------------------------------

def load_london_lad() -> gpd.GeoDataFrame:
    """London Local Authority Districts with an inner/outer label column."""
    gdf = gpd.read_file(LONDON_LAD_FILE)
    gdf["london_area"] = gdf["LAD24NM"].apply(
        lambda name: "Inner London" if name in INNER_LONDON_BOROUGHS else "Outer London"
    )
    return gdf


def load_london_lpa() -> gpd.GeoDataFrame:
    """London Local Planning Authority boundaries."""
    return gpd.read_file(LONDON_LPA_FILE)


def load_lsoa_boundaries(lsoa_codes: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load LSOA boundaries, optionally filtered to a list of LSOA codes."""
    gdf = gpd.read_file(LSOA_BOUNDARIES)
    if lsoa_codes is not None:
        gdf = gdf[gdf["LSOA21CD"].isin(lsoa_codes)]
    return gdf


# ---------------------------------------------------------------------------
# Census data
# ---------------------------------------------------------------------------

def load_census() -> pd.DataFrame:
    """Return a merged LSOA-level census DataFrame with three derived features:

    - ``percent_age_50_plus``
    - ``percent_occupation_1_2_3``
    - ``percent_owned_total``
    """
    age = pd.read_excel(CENSUS_AGE_FILE, sheet_name="2021")
    occ = pd.read_excel(CENSUS_OCCUPATION_FILE, sheet_name="2021")
    ten = pd.read_excel(CENSUS_TENURE_FILE, sheet_name="2021")

    # Derive features
    age_cols_50_plus = [
        "Aged 50 to 54", "Aged 55 to 59", "Aged 60 to 64",
        "Aged 65 to 69", "Aged 70 to 74", "Aged 75 to 79",
        "Aged 80 to 84", "Aged 85 and over",
    ]
    age["age_50_plus"] = age[age_cols_50_plus].sum(axis=1)
    age["percent_age_50_plus"] = (
        age["age_50_plus"] / age["All usual residents"] * 100
    )

    occ_cols_1_2_3 = [
        "1. Managers, directors and senior officials",
        "2. Professional occupations",
        "3. Associate professional and technical occupations",
    ]
    occ["occupation_1_2_3"] = occ[occ_cols_1_2_3].sum(axis=1)
    occ["percent_occupation_1_2_3"] = (
        occ["occupation_1_2_3"] / occ["All usual residents aged 16 and over in employment"] * 100
    )

    ten['owned_total'] = ten[['Owned outright', 'Owned with a mortgage or loan']].sum(axis=1)
    ten['percent_owned_total'] = ten['owned_total'] / ten['All Households'] * 100

    # Merge on LSOA code
    df = pd.merge(
        age[["LSOA code", "percent_age_50_plus"]],
        occ[["LSOA code", "percent_occupation_1_2_3"]],
        on="LSOA code",
        how="left",
    )
    df = pd.merge(
        df,
        ten[["LSOA code", "percent_owned_total"]],
        on="LSOA code",
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Combined / enriched datasets
# ---------------------------------------------------------------------------

def build_application_df(
    application_df: pd.DataFrame,
    pld_df: pd.DataFrame,
    comment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the application CSV with housing type and comment counts.

    Parameters
    ----------
    application_df:
        Output of :func:`load_planning_applications`.
    pld_df:
        Output of :func:`load_elasticsearch_planning`.
    comment_df:
        Output of :func:`load_comments`.

    Returns
    -------
    pd.DataFrame
        The application DataFrame with added 'housing_type' and
        'comment_count' columns, fully formatted.
    """
    # Merge housing type from Elasticsearch data
    df = pd.merge(
        application_df,
        pld_df[["lpa_app_no", "housing_type"]],
        on="lpa_app_no",
        how="left",
    )

    # Add comment counts
    app_count = (
        comment_df.groupby("application_id")["id"]
        .count()
        .reset_index()
        .rename(columns={"id": "comment_count"})
    )
    df = df.merge(app_count, left_on="lpa_app_no", right_on="application_id", how="left")
    df["comment_count"] = df["comment_count"].fillna(0).astype(int)
    df.drop(columns=["application_id"], inplace=True)

    df = ppf.format_df(df)
    return df


def build_comment_topic_df(
    comment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    pld_df: pd.DataFrame,
    topic_run_date: str,
) -> pd.DataFrame:
    """Merge comments with processed topic assignments and planning metadata.

    Parameters
    ----------
    comment_df:
        Output of :func:`load_comments`.
    topic_df:
        Output of :func:`load_topics` **after** topic post-processing
        (see ``topic_processing.py``).
    pld_df:
        Output of :func:`load_elasticsearch_planning`.
    topic_run_date:
        ISO date string used to isolate a single modelling run.

    Returns
    -------
    pd.DataFrame
        One row per comment, with topic lists, housing metadata, and
        planning application outcome merged in.
    """
    # Filter to the desired modelling run
    tp = topic_df[topic_df["add_date"] == topic_run_date].copy()

    # Merge comments with topics
    df = pd.merge(
        comment_df,
        tp[["comment_id", "topic_number", "probability"]],
        on="comment_id",
        how="left",
    )

    # Add a composite key for joining to planning data
    pld_df = pld_df.copy()
    pld_df["lpa_app_no_borough"] = pld_df["lpa_app_no"] + "_" + pld_df["borough"]
    df["lpa_app_no_borough"] = df["application_id"] + "_" + df["council"]

    df = pd.merge(
        df,
        pld_df[[
            "lpa_app_no_borough",
            # "outcome",
            "housing_type",
            "total_no_proposed_residential_units",
        ]],
        on="lpa_app_no_borough",
        how="left",
    )
    return df
