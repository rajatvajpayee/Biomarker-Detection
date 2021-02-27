# Project - Statistical analysis for discovering biomarkers

## Project Overview
The objective of this work is to analyse proteomics data for discovering biomarkers. The data is available in .mat format. The objective of this work is to reproduce  the work in the publication using python and other classification approaches. 

The following tasks have to be performed for analysing the data

- Formulate a classification problem and  build a classifier
- Perform cross-validation and perform systematic bootstrapping for 1000 for building an appropriate classifier.
- Compare the results with the published in the paper.

## Installing Dependencies
`requirements.txt` includes the packages required for this project. Run the below mentioned command in console -
```@python
pip install -r requirements.txt
```

## How to use
Run the below command to train the classifier - 
```@python
python src/compiled.py -f DATA_PATH -n ITERATIONS -s SAMPLE_SIZE -clf CLASSIFIER
```

To implement paper model, run the below command - 
```@python
python src/paper_implementation.py
``` 
