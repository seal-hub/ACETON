# ACETON 

This project contains an encoder-decoder model trained to automatically 
construct an energy test oracle for Android.

ACETON components:

* The sequence classification network using LSTM and Attention Layer (`LSTM_Attention.py`)
* The script to load the data returned from _Sequence Collector_ module (`load_dataset.py`) 
* The script for training the network using the training data (`train.py`) and testing the network using the test data (`test.py`)
* The list of network configurations (`config.py`)
* The script for tuning the sampling rate (`tune.py`) 
* The script for analysing attention weights and drawing heatmap of defect signatures (`feature_frequency.py`)

Runbook:

1) Make sure you have Python 3.7 and Pytorch
2) Put the dataset in the `data` directory and put the single tests in `data/tests` 
3) Check the model configs in `config.py`
4) Run `script.py` and follow its commands

The model implementation derived from the following links:

[1] https://github.com/prakashpandey9/Text-Classification-Pytorch

[2] https://github.com/philipperemy/keras-attention-mechanism/issues/14
