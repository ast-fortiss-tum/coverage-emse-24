# Replication package for the paper: "Can Optimization-Driven Search-Based Testing Effectively Cover Failure-Revealing Test Inputs? Insights from Two DL-Enabled Case Studies"
This repository holds the implementation and evaluation results of the study in the paper "Can Optimization-Driven Search-Based Testing Effectively Cover Failure-Revealing Test Inputs? Insights from Two DL-Enabled Case Studies".

# Results

## AVP

The results of the AVP Case Study can be found here: [\results-avp](results-avp)

The corresponding reference sets used for CID evaluation can be found here: [\ref-set\avp](ref-set\avp\oracle-variation) 
## MNIST

The results of the AVP Case Study can be found here: [\results-mnist](results-mnist)

The corresponding reference sets used for CID evaluation can be found here: [ref-set\mnist\oracle\variation](ref-set\mnist\oracle\variation) 

# Implementation

The implementation of the test case generation and evaluation is available in the folder [\code](code). The code can be used with the MNIST Case Study. For the AVP Case Study the SUT could not have been disclosed.
The case study has been implemented using the open-source search-based testing framework [OpenSBT](https://git.fortiss.org/opensbt).

## Preliminaries

Create a virtual environment and install all requirements by:

`pip install -r requirements.txt`

For troubleshooting related to MNIST dependencies we refer to the original implementation by Vincenzo et al. (DeepJanus):

## Test Case Generation

a) To start the generations of test cases for one single seed run:

`python analysis.py -r -p "/path/to/the/folder/with/runs/"`

b) To start the generations of test cases for multipled seeds use the script `run_analysis_seeds.sh`. Modify the seed number in line X to use different seeds for the evaluation. All seed number that have the expected label 5 are given in X.


## Evaluation

To start the evaluation:

`python analysis.py -p "/path/to/the/folder/with/runs/"`

The evaluation results will be written in the folder of the passed path of the runs.

# Authors


Lev Sorokin (sorokin@fortiss.org) \
Damir Safin (safin@fortiss.org)
