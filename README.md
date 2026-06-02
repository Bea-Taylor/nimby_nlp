# Neighbour NLP

Analysis code for the paper *'Hate thy neighbour'*, examining public representations submitted to UK planning authorities through natural language processing and spatial analysis.

## Background

When a planning application is submitted to a local planning authority (LPA) in England, members of the public are invited to submit representations — written comments that object to, support, or take a neutral stance on the proposal. For contentious applications this can generate hundreds of responses, creating a significant processing burden for planning officers who must read, categorise, and summarise all material received.

This repository provides the analysis pipeline for a corpus of planning representations collected across London boroughs. It applies topic modelling, sentiment analysis, spatial disaggregation, and demographic profiling to characterise patterns of public objection at scale.

## Repository Structure

```
notebooks/          # Analysis notebooks (run in order)
  01_descriptive_analysis.ipynb
  02_topic_analysis.ipynb
  03_sentiment_analysis.ipynb
  04_spatial_analysis.ipynb
  05_demographic_analysis.ipynb
  06_LAD_topic_maps.ipynb
functions/          # Shared utilities and configuration
data/               # Input data (see Data section below)
results/            # Outputs: tables, figures, parquet files
```

## Data

Raw representations were collected using [comment_mill](https://github.com/AI4CI-smart-cities/comment_mill). Pre-processed data and NLP model outputs are hosted on Zenodo:

> Dataset: https://zenodo.org/records/19567801

The analysis also draws on 2021 Census data (age, occupation, tenure at LSOA level) and ONS/DLUHC geospatial boundaries for London Local Authority Districts and LPAs.

## NLP Models

Topic modelling and sentiment classification models were generated using [comment_crunch](https://github.com/AI4CI-smart-cities/comment_crunch). The outputs — topic assignments and sentiment scores per comment — are provided as pre-computed CSVs in `data/` and can be reproduced independently via that repository.

Core dependencies include `BERTopic`, `sentence-transformers`, `transformers`, and `geopandas`. See `environment.yml` for the full specification.

## Publications

Conference paper: https://zenodo.org/records/19567801

Journal paper: *in preparation*

## Attribution

Developed by the [AI4CI Smart Cities](https://github.com/AI4CI-smart-cities) project at the [Centre for Advanced Spatial Analysis (CASA)](https://www.ucl.ac.uk/bartlett/casa), UCL.
