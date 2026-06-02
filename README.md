# ***Neighbour NLP***

The code inside this repo generates the analysis for the paper 'Hate thy neighbour'. 

## So what's it all about? 

In the UK, following the submission of a planning application to the local council or local planning authority (LPA), there is a phase whereby members of the local community are allowed to comment on the application. In the planning community these comments are called representations. Typically councils ask community repsondents to classify whether their feedback 'objects', 'supports' or takes a 'neutral' stance toward the planning application. 

More controversial projects might result in 100s of representations - which can be time consuming for councils to process, since planning opfficers have a responsibility to read-through, categorise and sumamrise any comments received. As part of this process they also classify whether obejctions are material or non-material. 

This work uses data collected [*comment_mill*](https://github.com/AI4CI-smart-cities/comment_mill). This repo provides the code for analysing this data.

## The data 
You can see the code I wrote for web-scraping the data in [comment_summariser](). 

## ML models 
The code to generate the ML models can also be found in [comment_crunch](). This creates the NLP analysis - specifically the topic modelling and sentiment analysis. 

### Sharing the work

This tool was developed by AI4CI Smart Cities project in the [Centre for Advanced Spatial Analysis (CASA)](https://www.ucl.ac.uk/bartlett/casa).
