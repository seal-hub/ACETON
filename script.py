#!/usr/bin/env python3

import sys

import load_dataset
import train
import test
import tune
import feature_frequency
import config


def print_runbook():
    print('** ACETON runbook **')
    print('use one of the following arguments:')
    print("load <dataset_name>: to load a dataset and split it to test and train sets")
    print(
        "train <dataset_name> <model_name>: to train a model on the train set of a dataset and save as <model_name>.pt")
    print(
        "test <model_name> <dataset_name> <attention_weights_name>: to evaluate a model on the test set of a dataset and save the attention weights as <attention_weights_name>.csv")
    print("test_one <model_name> <test_name.txt>: to predict the label of a specific test file using a model")
    print("test_ex <model_name> <dataset_name> <subcat_num>: to evaluate a model on an excluded subcategory")
    print("tune <dataset_name>: to get the best sampling rate")
    print(
        "heatmap <attention_name> <dataset_name> <feature_frequency_name> <plots_directory:optional>: to create a heatmap of aggregated features around attended SV. \n"
        "        It also returns the feature frequency file and can save the attention plots in the specified directory ")
    print("unseen <dataset_name> <excluded_subcat>: evaluate the predictability of the model on an unseen subcategory")


if len(sys.argv) == 1 or sys.argv[1] == 'help':
    print_runbook()


elif sys.argv[1] == 'load':
    if len(sys.argv) == 3:
        load_dataset.load(sys.argv[2])
        print('dataset loaded in ' + "Loaded_" + sys.argv[2])
    else:
        print_runbook()

elif sys.argv[1] == 'train':
    if len(sys.argv) == 4:
        train.run_train(sys.argv[2], sys.argv[3])
    else:
        print_runbook()

elif sys.argv[1] == 'test':
    if len(sys.argv) == 5:
        test.run_test(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print_runbook()

elif sys.argv[1] == 'test_one':
    if len(sys.argv) == 4:
        test.run_test_one(sys.argv[2], sys.argv[3])
    else:
        print_runbook()

elif sys.argv[1] == 'tune':
    if len(sys.argv) == 3:
        tune.tune_sampline_rate(sys.argv[2])
    else:
        print_runbook()

elif sys.argv[1] == 'heatmap':
    if len(sys.argv) == 6:
        feature_frequency.calculate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) == 5:
        feature_frequency.calculate(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print_runbook()

elif sys.argv[1] == 'unseen':
    if len(sys.argv) == 4:
        config.num_epochs = 6
        train.run_train(sys.argv[2], "unseen"+sys.argv[3], sys.argv[3])
    else:
        print_runbook()

else:
    print_runbook()
