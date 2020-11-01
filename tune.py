import numpy as np
from random import randrange
import config
from LSTM_Attention import AttentionModel
import torch
import pickle
from torch.nn import functional as F


def tune_sampline_rate(dataset_name):

    # load dataset
    with open('data/Loaded-' + dataset_name + '/data_train.dump', 'rb') as dump_data_file:
        X_train = pickle.load(dump_data_file)

    label_train = torch.from_numpy(
        np.loadtxt(open('data/Loaded-' + dataset_name + '/label_train.csv', "rb"), delimiter=",",
                   skiprows=1))

    x_train = torch.from_numpy(X_train)
    y_train = label_train.permute(1, 0)

    # investigating the best sampling size
    x_train_copy = x_train
    candidate_time_steps = [2, 8, 16, 32, 64, 128]
    recall_list = []
    for time_step in candidate_time_steps:
        indices = []
        for i in range(time_step - 1):
            indices.append(int(i * config.num_steps / (time_step - 1)))
        indices.append(config.num_steps - 1)

        sampled_steps = indices
        x_train = x_train_copy[:, sampled_steps, :]
        k = 10

        kfold_indicies = cross_validation_split(x_train.shape[0], k)
        kfold_recall = np.zeros(k)

        for kk in range(k):
            model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=config.batch_size,
                                   output_dim=config.num_classes, num_layers=config.num_layers,
                                   max_seq_len=config.num_steps)
            cv_x_test = x_train[kfold_indicies[kk], :, :]
            cv_y_test = y_train[:, kfold_indicies[kk]]
            cv_train_indicies = []
            for i in range(k):
                if i == k:
                    continue
                cv_train_indicies.extend(kfold_indicies[i])

            cv_x_train = x_train[cv_train_indicies, :, :]
            cv_y_train = y_train[:, cv_train_indicies]

            train(model, cv_x_train, cv_y_train, calculate_weights(cv_y_train))

            best_model = AttentionModel(config.num_features, config.hidden_dimension, batch_size=1,
                                        output_dim=config.num_classes, num_layers=config.num_layers,
                                        max_seq_len=config.num_steps)
            best_model.load_state_dict(torch.load("cv.pt"))  # config.loss_model_name
            best_model.eval()

            confusion_matrix = np.zeros((config.num_classes, config.num_classes))
            for j in range(cv_x_test.shape[0]):

                test_target = cv_y_test[0][j]
                out, _ = best_model(cv_x_test[j].unsqueeze(0), )
                predicted = torch.max(out.squeeze(), 0)[1].item()

                confusion_matrix[predicted][int(test_target.item())] += 1

            recall = [confusion_matrix[z][z] / confusion_matrix.sum(0)[z] for z in
                      range(config.num_classes)]  # recall: diag/sumCol

            kfold_recall[kk] = np.average(recall)
            print('timestep', time_step, ', fold:', kk)
            print("avg_recall", np.average(recall))
        recall_list.append(np.average(kfold_recall))

    print(recall_list)
    recall_array = np.asarray(recall_list)
    recall_array[np.isnan(recall_array)]=0
    print('Best sampling rate according to average recall:', candidate_time_steps[recall_array.argmax()])


# Split a dataset into k folds
def cross_validation_split(dataset_size, folds=10):
    dataset_split_index = list()
    fold_size = int(dataset_size / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(dataset_size)
            fold.append(index)
        dataset_split_index.append(fold)
    return dataset_split_index


def calculate_weights(y_train):
    unique_labels, counts_elements = np.unique(y_train[0], return_counts=True)
    print("labels and counts: ", unique_labels, counts_elements)
    weight = torch.full((1, len(unique_labels)), 1)
    if config.weighted_loss:
        weight = torch.zeros(len(unique_labels))
        for q in range(len(unique_labels)):
            weight[q] = min(counts_elements) / counts_elements[q]
    return weight


def train(model, all_x_train, all_y_train, weight):

    criterion = F.cross_entropy
    optimiser = torch.optim.Adam(model.parameters())

    train_hist = np.zeros(config.num_epochs)
    val_hist = np.zeros(config.num_epochs)
    rec_hist = np.zeros(config.num_epochs)
    least_loss = 10

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
            validation_loss += criterion(y_pred, val_target.squeeze(), weight=weight)
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
            torch.save(model.state_dict(), "cv.pt")

        if patience < 0 or val_hist[t] < config.early_stop_threshold:
            print('Early terminated in epoch ', t)
            break


if __name__ == '__main__':
    tune_sampline_rate('Dataset-12')
