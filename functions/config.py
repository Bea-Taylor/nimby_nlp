"""
config.py
---------
Central configuration for the planning comments analysis project.
Edit this file to change data paths or visual styles.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Input datasets
COMMENTS_CSV        = DATA_DIR / "comments.csv"
COMMENTS_TOPICS_CSV = DATA_DIR / "comments_with_topics.csv"
TOPIC_NAMES_CSV     = DATA_DIR / "topic_names.csv"

# Pre-computed planning datasets (no CSV equivalent — kept in results/)
PLD_FILE          = RESULTS_DIR / "pld.parquet"
APPLICATIONS_FILE = RESULTS_DIR / "applications.parquet"

# Geospatial boundaries
LONDON_LAD_FILE = DATA_DIR / "london_geos" / "Local_Authority_Districts_May_2024_London.geojson"
LONDON_LPA_FILE = DATA_DIR / "london_geos" / "local-planning-authority-london.geojson"
LSOA_BOUNDARIES = DATA_DIR / "london_geos" / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-4299016806856585929.geojson"

# Census data
CENSUS_AGE_FILE        = DATA_DIR / "2021_census" / "Five year age bands.xlsx"
CENSUS_OCCUPATION_FILE = DATA_DIR / "2021_census" / "occupation.xlsx"
CENSUS_TENURE_FILE     = DATA_DIR / "2021_census" / "tenure - households.xlsx"

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

COLOUR_PRIMARY   = "#894e9e"
COLOUR_SECONDARY = "#abc766"
COLOUR_ACCENT    = "#e16fca"
COLOUR_YELLOW    = "#f9dd73"
COLOUR_TEAL      = "#2e6260"
COLOUR_BROWN     = "#a9746e"
COLOUR_GREY      = "#9f9f9f"

STANCE_PALETTE = {
    "Supports": "#b8cd73",
    "Objects":  "#cf4242",
    "Neutral":  "#426acf",
}

TOPIC_COLOURS = [
    "#336589", "#aad874", "#fba337", "#b63838", "#a373d0",
    "#865349", "#F2F527", "#494A2A", "#df51b4", "#7d6565",
    "#1cc6d9", "#453D70", "#bcbd22", "#09A31B", "#91e0b0",
    "#a1c7e3", "#d1e7c3", "#e3d1bc", "#b48989", "#4e0f88",
    "#D12E0E", "#B4B489", "#EDEEE0", "#df9fcc", "#e58888",
    "#a8dadf", "#7A6CC8", "#A7A70A", "#A1D6A7", "#27c968",
]

HOUSING_TYPE_COLOURS = {
    "Market housing":           "#894e9e",
    "Mixed affordable housing": "#abc766",
    "Mixed social housing":     "#e16fca",
    "Social housing":           "#f9dd73",
    "Self-build housing":       "#2e6260",
    "Other":                    "#a9746e",
}

HOUSING_ORDER = [
    "Market housing",
    "Mixed affordable housing",
    "Mixed social housing",
    "Social housing",
    "Self-build housing",
    "Other",
]

INNER_LONDON_BOROUGHS = [
    "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham",
    "Islington", "Kensington and Chelsea", "Lambeth", "Lewisham",
    "Southwark", "Tower Hamlets", "Wandsworth", "Westminster",
    "City of London", "Newham",
]
