"""
config.py
---------
Central configuration for the planning comments analysis project.
Edit this file to change data paths, topic groupings, or visual styles.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (all relative to the repo root so the project is portable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repo root

DATA_DIR          = ROOT / "data"
RESULTS_DIR       = ROOT / "results"
FIGURES_DIR       = RESULTS_DIR / "figures"
MODEL_OUTPUTS_DIR = ROOT / "model_outputs"

# Raw data files
APPLICATION_CSV   = DATA_DIR / "PLD_application_ids" / "all_since21_cleaned.csv"

# Geospatial boundaries
LONDON_LAD_FILE   = DATA_DIR / "london_geos" / "Local_Authority_Districts_May_2024_London.geojson"
LONDON_LPA_FILE   = DATA_DIR / "london_geos" / "local-planning-authority-london.geojson"
LSOA_BOUNDARIES   = DATA_DIR / "london_geos" / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-4299016806856585929.geojson"

# Census data
CENSUS_AGE_FILE       = DATA_DIR / "2021_census" / "Five year age bands.xlsx"
CENSUS_OCCUPATION_FILE= DATA_DIR / "2021_census" / "occupation.xlsx"
CENSUS_TENURE_FILE    = DATA_DIR / "2021_census" / "tenure - households.xlsx"

# Topic model outputs
FINETUNED_MODEL_DIR   = MODEL_OUTPUTS_DIR / "finetuned_objection"
BASELINE_MODEL_DIR    = MODEL_OUTPUTS_DIR / "not_tuned_objection"
BERTOPIC_MODEL_PATH   = MODEL_OUTPUTS_DIR / "topic_model" / "bertopic_less_topics" / "full_model" / "object_model"

# ---------------------------------------------------------------------------
# Elasticsearch / database
# ---------------------------------------------------------------------------
ES_QUERY_PARAMS = dict(
    min_res_units = 1,
    since_year    = "01/01/2021",
    to_year       = "01/05/2025",
)

# The date used to isolate a single topic-modelling run
TOPIC_RUN_DATE = "2026-01-24"

# ---------------------------------------------------------------------------
# Topic post-processing
# ---------------------------------------------------------------------------

# Topics to discard entirely (noise / artefacts from BERTopic)
BAD_TOPICS = [-1, 0, 13, 24, 34, 39, 40, 43, 45, 49, 66, 69, 81, 88, 89, 90, 95]

# Topics that are semantically equivalent – the first id in each list is the
# canonical representative; all others are merged into it.
SAME_TOPICS = {
    "impact on parking":              [1, 15, 30, 35, 42, 47, 48, 63],
    "loss of gardens":                [2, 41],
    "out of character":               [3, 56, 77, 91],
    "construction access":            [4],
    "too tall":                       [5, 23],
    "impact on heritage and conservation": [6, 31, 46, 84],
    "impact on amenities":            [7, 86, 27],
    "loss of privacy":                [8],
    "loss of light":                  [9],
    "wrong housing type":             [10, 68],
    "overdevelopment":                [11, 22],
    "noise pollution":                [12, 32, 37, 55],
    "loss of trees":                  [14, 57],
    "traffic impact":                 [16, 44],
    "impact on wildlife":             [17],
    "drainage and flooding":          [18],
    "loss of views":                  [19],
    "safety concerns":                [20],
    "air quality":                    [21],
    "demand for affordable housing":  [25],
    "impact on schools":              [26],
    "boundary disputes":              [28],
    "loss of sunlight":               [29],
    "impact on infrastructure":       [33],
    "impact on local character":      [36],
    "loss of employment":             [38],
    "impact on public transport":     [50],
    "insufficient community consultation": [51],
    "environmental impact":           [52],
}

# Derived lookup: every topic id → its canonical id
TOPIC_MAP: dict[int, int] = {
    t: topics[0]
    for topics in SAME_TOPICS.values()
    for t in topics
}

# Derived lookup: canonical id → human-readable group name
TOPIC_GROUP_MAP: dict[int, str] = {
    topics[0]: group_name
    for group_name, topics in SAME_TOPICS.items()
}

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

# Primary brand colours
COLOUR_PRIMARY   = "#894e9e"   # purple
COLOUR_SECONDARY = "#abc766"   # green
COLOUR_ACCENT    = "#e16fca"   # pink
COLOUR_YELLOW    = "#f9dd73"
COLOUR_TEAL      = "#2e6260"
COLOUR_BROWN     = "#a9746e"
COLOUR_GREY      = "#9f9f9f"

# Palette for commenter stance
STANCE_PALETTE = {
    "Supports": "#426acf",
    "Objects":  "#cf4242",
    "Neutral":  "#b8cd73",
}

# Per-topic colours (rotated over the list when there are more topics than colours)
TOPIC_COLOURS = [
    "#336589", "#aad874", "#fba337", "#b63838", "#a373d0",
    "#865349", "#F2F527", "#494A2A", "#df51b4", "#7d6565",
    "#1cc6d9", "#453D70", "#bcbd22", "#09A31B", "#91e0b0",
    "#a1c7e3", "#d1e7c3", "#e3d1bc", "#b48989", "#4e0f88",
    "#D12E0E", "#B4B489", "#EDEEE0", "#df9fcc", "#e58888",
    "#a8dadf", "#7A6CC8", "#A7A70A", "#A1D6A7", "#27c968",
]

# Housing type colours
HOUSING_TYPE_COLOURS = {
    "Market housing":          "#894e9e",
    "Mixed affordable housing":"#abc766",
    "Mixed social housing":    "#e16fca",
    "Social housing":          "#f9dd73",
    "Self-build housing":      "#2e6260",
    "Other":                   "#a9746e",
}

HOUSING_ORDER = [
    "Market housing",
    "Mixed affordable housing",
    "Mixed social housing",
    "Social housing",
    "Self-build housing",
    "Other",
]

# Inner-London boroughs used for inner/outer splits
INNER_LONDON_BOROUGHS = [
    "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham",
    "Islington", "Kensington and Chelsea", "Lambeth", "Lewisham",
    "Southwark", "Tower Hamlets", "Wandsworth", "Westminster",
    "City of London", "Newham",
]
