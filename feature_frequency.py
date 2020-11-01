import csv
import math
import os
import numpy as np
import config
import pickle
import pylab as plt
import torch

from config import gamma


def calculate(attention_name, dataset_name, frequency_name, plot_directory=None):
    with open('data/Loaded-' + dataset_name + '/data_test.dump', 'rb') as dump_data_file:
        X_test = pickle.load(dump_data_file)
    label_test = torch.from_numpy(
        np.loadtxt(open('data/Loaded-' + dataset_name + '/label_test.csv', "rb"), delimiter=",", skiprows=1))

    x_test = torch.from_numpy(X_test)
    y_test = label_test.permute(1, 0)

    feature_frequency_file = open(frequency_name+".csv", "a")
    freq_header = ['cat', 'dist']
    for k in range(config.num_features):
        freq_header.append('f' + str(k))
    freq_writer = csv.writer(feature_frequency_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if os.stat(frequency_name+".csv").st_size == 0:
        freq_writer.writerow(freq_header)
    attention = np.loadtxt(open(attention_name+'.csv', "rb"), delimiter=",", skiprows=1, usecols=[0] + list(range(3, 132)))
    file_names = np.genfromtxt(open(attention_name + '.csv', "rb"), delimiter=",", skip_header=1, usecols=[1], dtype='str')
    print(attention.shape)
    frequency = np.zeros((len(config.subcategories), 3, config.num_features))

    for i in range(attention.shape[0]):
        attention_weights = attention[i, 2:]
        test_index = int(attention[i, 0])
        injected_index = int(attention[i, 1])

        pk, length_pk = peak_beginning(attention_weights.squeeze().tolist())
        estimate = pk

        if plot_directory is not None:
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)
            plt.plot(attention_weights)
            plt.savefig(plot_directory + "/" + file_names[i] + ".png")
            plt.clf()

        freq = localize_features(estimate, x_test[test_index, :, :])
        cat_target = y_test[1][test_index]

        frequency[int(cat_target.item())-1] += freq

    subcat_dict = dict(config.subcategories)
    frequency_sum = np.zeros((len(config.subcategories), config.num_features))
    for i in range(22):
        if np.max(frequency[i]) == 0:
            continue
        for j in range(3):
            frequency_sum[i] += math.pow(gamma, j) * frequency[i, j]
            row = [list(subcat_dict.keys())[list(subcat_dict.values()).index(i+1)], "["+str(1+((j+1)*3))+"+]"]
            row.extend(frequency[i][j])
            freq_writer.writerow(row)
        frequency_sum[i] = frequency_sum[i] / np.max(frequency_sum[i])

    # AS REQUESTED!
    frequency_sum[:7, 0] = 0
    frequency_sum[8:, 0] = 0
    frequency_sum[:, 4] = 0
    frequency_sum[:, 2] = 0
    frequency_sum[:, 5] = 0
    frequency_sum[10:11, 1:9] = frequency_sum[10:11, 1:9]/10 # fade lifecycle features
    frequency_sum[14, 1:9] = frequency_sum[14, 1:9] / 10  # fade lifecycle features
    frequency_sum[17, 1:9] = frequency_sum[17, 1:9] / 10  # fade lifecycle features

    im = plt.imshow(frequency_sum, cmap='jet', interpolation='nearest')
    plt.colorbar(im, orientation='horizontal')
    plt.xticks(np.arange(84, step=4))
    plt.yticks(np.arange(23), subcat_dict.keys())
    plt.show()

    feature_frequency_file.close()


def localize_features(estimate, x_test):
    freq = np.zeros((3, config.num_features))
    dif_ind = []
    dif_big_ind = []
    dif_biggest_ind = []
    for feature in range(84):
        myset = set(x_test[ max(0, estimate):min(127, estimate + 4), feature].tolist())
        biggerset = set(x_test[max(0, estimate):min(127, estimate + 7), feature].tolist())
        biggestset = set(x_test[max(0, estimate):min(127, estimate + 10), feature].tolist())
        if len(myset) > 1:
            freq[0][feature] += 1
            dif_ind.append(feature)
        if len(biggerset) > 1:
            if freq[0][feature] == 0:
                freq[1][feature] += 1
                dif_big_ind.append(feature)
        if len(biggestset) > 1:
            if freq[0][feature] == 0 and freq[1][feature] == 0:
                freq[2][feature] += 1
                dif_biggest_ind.append(feature)
    diff = "[+4]:" + str(dif_ind) + "*[+7]:" + str(dif_big_ind) + "*[+10]:" + str(dif_biggest_ind)
    return freq, diff


def peak_beginning(attention_weights):
    tallest = -1
    index = -1
    for j in range(len(attention_weights)-1):
        if attention_weights[j + 1] < attention_weights[j]:
            continue
        start = j
        while j<len(attention_weights)-1 and attention_weights[j+1] > attention_weights[j]:
            j += 1
        if (attention_weights[j] - attention_weights[start]) > tallest:
            tallest = (attention_weights[j] - attention_weights[start])
            index = start
    return index, tallest


if __name__ == '__main__':
    calculate('my_attention', "Dataset-12", "my_freq", "plots")