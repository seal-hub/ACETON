import csv
import os
import pickle
import numpy as np
import torch

import feature_frequency
from LSTM_Attention import AttentionModel
import config
import matplotlib.pyplot as plt
import re


def run_test(model_name, dataset_name, attention_name):
    subcat_dict = dict(config.subcategories)
    with open('data/Loaded-' + dataset_name + '/data_test.dump', 'rb') as dump_data_file:
        X_test = pickle.load(dump_data_file)
    label_test = torch.from_numpy(
        np.loadtxt(open('data/Loaded-' + dataset_name + '/label_test.csv', "rb"), delimiter=",", skiprows=1))

    x_test = torch.from_numpy(X_test)
    y_test = label_test.permute(1, 0)

    print("testing on ", x_test.shape[0], " samples...")

    best_model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=1,
                                output_dim=config.num_classes, num_layers=config.num_layers,
                                max_seq_len=config.num_steps)
    best_model.load_state_dict(torch.load(model_name+".pt"))
    best_model.eval()

    confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
    conf_cat = torch.zeros(config.num_classes, len(config.subcategories)+1)
    res = open(attention_name+".csv", "a")
    res_writer = csv.writer(res, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    att_header = ['index','cat-file', 'predicted_label', 'injected_index' ]
    for k in range(128):
        att_header.append('w' + str(k))
    if os.stat(attention_name+".csv").st_size == 0:
        res_writer.writerow(att_header)
    for j in range(x_test.shape[0]):

        test_target = y_test[0][j]
        cat_target = y_test[1][j]
        loc_target = y_test[2][j]

        out, attention_weights = best_model(x_test[j].unsqueeze(0), )

        predicted = torch.max(out.squeeze(), 0)[1].item()
        if predicted > 0:
            cat_name = "pass" if cat_target == 0 else list(subcat_dict.keys())[list(subcat_dict.values()).index(int(cat_target.item()))]
            row = [str(j), cat_name + "-" + str(int(y_test[3][j].item())), int(predicted), str(int(loc_target.item()))]
            row.extend(attention_weights.squeeze().tolist())
            res_writer.writerow(row)
        confusion_matrix[predicted][int(test_target.item())] += 1
        conf_cat[predicted][int(cat_target.item())] += 1

    precision = [confusion_matrix[z][z] / torch.sum(confusion_matrix, 1)[z] for z in
                 range(config.num_classes)]  # precision: diag/sumRow
    recall = [confusion_matrix[z][z] / torch.sum(confusion_matrix, 0)[z] for z in
              range(config.num_classes)]  # recall: diag/sumCol
    accuracy = np.trace(confusion_matrix.numpy()) / np.sum(confusion_matrix.numpy())

    print('confusion_matrix:\n', confusion_matrix)
    print('precision:', precision)
    print('recall:', recall)
    print('accuracy:', accuracy)
    print('confusion_matrix_per_subcat:\n', conf_cat)
    res.close()


def run_test_one(model_name, file):
    x_test = test_loader(file)
    best_model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=1,
                                output_dim=config.num_classes, num_layers=config.num_layers,
                                max_seq_len=config.num_steps)
    best_model.load_state_dict(torch.load(model_name + ".pt"))
    best_model.eval()
    out, attention_weights = best_model(torch.from_numpy(x_test).unsqueeze(0), )
    predicted = torch.max(out.squeeze(), 0)[1].item()
    print("label:", predicted)

    # plot attention weights
    plt.plot([i for i in attention_weights.squeeze().tolist()])
    plt.title("Attention weights")
    plt.show()

    # hint on relevant features
    estimate, _ = feature_frequency.peak_beginning(attention_weights)
    _, delta = feature_frequency.localize_features(estimate, x_test)
    print(delta)


def test_loader(filename):
    f = open("data/tests/" + filename)
    lines = f.readlines()

    time_step = 0
    if len(lines) == 129:
        lines.append(lines[128] + '\n')

    X_test = np.zeros((config.num_steps, config.num_features))
    for line in lines[2:129]:
        line = "".join(line.split())

        line_tokens = re.split("[, \[\]\W]+", line[1:-1])

        X_test[time_step, :] = list(map(int, line_tokens))

        time_step += 1
    return X_test


if __name__ == '__main__':
    run_test("pretrained_model", "Dataset-12", "my_attention")