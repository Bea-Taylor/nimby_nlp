# ***comment crunch***

The code inside this repo provides the pipeline for processing comments from planning applications, and generating reports and summary statistics.

## So what's it all about? 

In the UK, following the submission of a planning application to the local council or local planning authority (LPA), there is a phase whereby members of the local community are allowed to comment on the application. In the planning community these comments are called representations. Typically councils ask community repsondents to classify whether their feedback 'objects', 'supports' or takes a 'neutral' stance toward the planning application. 

More controversial projects might result in 100s of representations - which can be time consuming for councils to process, since planning opfficers have a responsibility to read-through, categorise and sumamrise any comments received. As part of this process they also classify whether obejctions are material or non-material. 

This work uses data collected [*comment_mill*](https://github.com/AI4CI-smart-cities/comment_mill). This repo provides the code for processing this data, starting with developing a fine-tuned domain specific transformer model. This is then used for downstream tasks including topic modelling and sentiment analysis. 

## How do I implement it? 

Coming soon!

### Sharing the work

This tool was developed by AI4CI Smart Cities project in the [Centre for Advanced Spatial Analysis (CASA)](https://www.ucl.ac.uk/bartlett/casa).


#### Docker

If you wish to install and run via a Jupyter notebook in Docker:

Build the image:

```
docker build -t "comment_crunch" .
```

Run the image:
```
docker run -p 8888:8888 comment_crunch
```

Then navigate in your browser to:

[127.0.0.1:8888](127.0.0.1:8888)