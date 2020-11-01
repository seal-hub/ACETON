import torch
import pickle
import config
import numpy as np
from torch.nn import functional as F
import os
from random import randrange
from LSTM_Attention import AttentionModel


def run_train(dataset_name, model_name, excluded_subcat = None):

    with open('data/Loaded-' + dataset_name + '/data_train.dump', 'rb') as dump_data_file:
        X_train = pickle.load(dump_data_file)

    label_train = torch.from_numpy(
        np.loadtxt(open('data/Loaded-' + dataset_name + '/label_train.csv', "rb"), delimiter=",",
                   skiprows=1))

    x_train = torch.from_numpy(X_train)
    y_train = label_train.permute(1, 0)

    if excluded_subcat is not None:
        excluded_subcat = int(excluded_subcat)
        subcat_dict = dict(config.subcategories)
        ind = label_train[:, 1] != excluded_subcat
        excluded_ind = label_train[:, 1] == excluded_subcat

        new_y_train = y_train[:, ind]
        new_x_train = x_train[ind]

        x_exc = x_train[excluded_ind]
        y_exc = y_train[:, excluded_ind]

        x_train = new_x_train
        y_train = new_y_train
        print('excluding category: ', list(subcat_dict.keys())[list(subcat_dict.values()).index(excluded_subcat)], ' total num: ', x_exc.shape[0])

    indices = []
    for i in range(config.sampling_rate - 1):
        indices.append(int(i * config.num_steps / (config.sampling_rate - 1)))
    indices.append(config.num_steps - 1)

    x_train = x_train[:, indices, :]

    model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=config.batch_size,
                           output_dim=config.num_classes, num_layers=config.num_layers, max_seq_len=config.num_steps)

    weight = calculate_weights(y_train)
    print('weights: ', weight.tolist())

    train(model, x_train, y_train, weight, model_name)

    if excluded_subcat is not None:
        best_model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=1,
                                    output_dim=config.num_classes, num_layers=config.num_layers,
                                    max_seq_len=config.num_steps)
        best_model.load_state_dict(torch.load("unseen"+str(excluded_subcat)+".pt"))
        best_model.eval()
        print('results on the excluded category:')
        confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        for j in range(x_exc.shape[0]):
            test_target = y_exc[0][j]
            out, attention_weights = best_model(x_exc[j].unsqueeze(0), )
            predicted = torch.max(out.squeeze(), 0)[1].item()
            confusion_matrix[predicted][int(test_target.item())] += 1

        os.remove("unseen"+str(excluded_subcat)+".pt")
        print('recall:', np.trace(confusion_matrix.numpy()) / np.sum(confusion_matrix.numpy()))


def calculate_weights(y_train):
    unique_labels, counts_elements = np.unique(y_train[0], return_counts=True)
    print("labels and counts: ", unique_labels, counts_elements)
    weight = torch.full((1, len(unique_labels)), 1)
    if config.weighted_loss:
        weight = torch.zeros(len(unique_labels))
        for q in range(len(unique_labels)):
            weight[q] = min(counts_elements) / counts_elements[q]
    return weight


def train(model, all_x_train, all_y_train, weight, model_name):

    criterion = F.cross_entropy
    optimiser = torch.optim.Adam(model.parameters())

    train_hist = np.zeros(config.num_epochs)
    val_hist = np.zeros(config.num_epochs)
    rec_hist = np.zeros(config.num_epochs)
    least_loss = 1

    indices = list(range(all_x_train.shape[0]))
    split = int(np.floor(config.validation_split * all_x_train.shape[0]))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    x_val = all_x_train[val_indices]
    y_val = all_y_train[:, val_indices]
    x_train = all_x_train[train_indices]
    y_train = all_y_train[:, train_indices]

    unique_labels, counts_elements = np.unique(y_train[0], return_counts=True)
    print("trn, labels and counts: ", unique_labels, counts_elements)
    unique_labels, counts_elements = np.unique(y_val[0], return_counts=True)
    print("val, labels and counts: ", unique_labels, counts_elements)

    patience = 1

    for t in range(config.num_epochs):

        n_batches = int(len(x_train) / config.batch_size)
        for i in range(n_batches):
            model.train()
            model.zero_grad()
            last_trn_idx = (i + 1) * config.batch_size

            target = torch.autograd.Variable(y_train[0][i * config.batch_size:last_trn_idx]).long()

            y_pred, _ = model(x_train[i * config.batch_size:last_trn_idx])

            loss = criterion(y_pred, target.squeeze(), weight=weight)
            train_hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()

        n_val_batches = int(len(x_val) / config.batch_size)

        confusion_matrix = np.zeros((len(config.categories)+1, len(config.categories)+1))

        validation_loss = 0
        for j in range(n_val_batches):
            last_idx = min((j + 1) * config.batch_size, x_val.shape[0])
            val_target = torch.autograd.Variable(y_val[0][j * config.batch_size:last_idx]).long().squeeze()
            y_pred, _ = model(x_val[j * config.batch_size:last_idx, ])
            validation_loss +=criterion(y_pred, val_target.squeeze(), weight=weight)
            for i in range(val_target.squeeze().shape[0]):
                confusion_matrix[torch.max(y_pred, 1)[1].view(val_target.size()).data[i]][val_target.data[i]] += 1

        validation_loss = validation_loss / n_val_batches
        if t > 1 and validation_loss.item() > val_hist[t-1]:
            patience -= 1
        else:
            patience = 1

        recall = [confusion_matrix[z][z] / confusion_matrix.sum(0)[z] for z in
                  range(config.num_classes)]  # recall: diag/sumCol
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        val_hist[t] = validation_loss.item()
        rec_hist[t] = np.average(recall)

        print("Epoch ", t, "CE: ", loss.item(), "Val_Acc: ", accuracy, "Val_Loss: ", validation_loss, 'Val_avg_recall:', np.average(recall))
        if validation_loss < least_loss:
            least_loss = validation_loss
            torch.save(model.state_dict(), model_name+".pt")

        if patience < 0 or val_hist[t] < config.early_stop_threshold:
            print('Early terminated in epoch ', t)
            break


if __name__ == '__main__':
    run_train("Dataset-12", "my_model")
