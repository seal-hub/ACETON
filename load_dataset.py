import os
import random
import re
import numpy as np
import pickle
import csv
import config
from config import categories, subcategories


def load(dataset_name):

    test_size = 0
    train_size = 0
    for path, directories, files in os.walk("data/"+dataset_name):
        if len(path.split("/")) != 5:  # data/dataset_name/cat_name/sub_cat_name/fail_pass
            continue

        current_test_size = int(0.1 * (len(files) - files.count(".DS_Store")))
        test_size += current_test_size
        train_size += len(files) - files.count(".DS_Store") - current_test_size

    print('loading dataset of ', train_size, " train sample, ", test_size, " test sample...")

    cat_dict = dict(categories)
    subcat_dict = dict(subcategories)
    if not os.path.exists('data/Loaded-'+dataset_name):
        os.makedirs('data/Loaded-'+dataset_name)

    X_train = np.zeros((train_size, config.num_steps, config.num_features))
    label_train = open('data/Loaded-'+dataset_name+'/label_train.csv','a')
    X_test = np.zeros((test_size, config.num_steps, config.num_features))
    label_test = open('data/Loaded-'+dataset_name+'/label_test.csv','a')

    label_train_writer = csv.writer(label_train, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    label_test_writer = csv.writer(label_test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    label_train_writer.writerow(['label', 'category', 'index', 'name'])
    label_test_writer.writerow(['label', 'category', 'index', 'name'])

    test_sample = 0
    train_sample = 0

    for path, directories, files in os.walk("data/"+dataset_name):
        if len(path.split("/")) != 5:
            continue
        random.shuffle(files)
        tokens = path.split("/")
        label = cat_dict[tokens[2]] if tokens[-1] == 'Fail' else 0
        cat = subcat_dict[tokens[3]] if tokens[-1] == 'Fail' else 0
        test_size = int(0.1 * (len(files) - files.count(".DS_Store")))
        test_counter = 0
        for filename in files:
            if filename == ".DS_Store":
                continue
            if test_counter < test_size:
                print('test')
                print('processing: '+path+"/"+filename+"/test_size:"+str(test_size))
                f = open(path+"/"+filename)
                lines = f.readlines()
                index = int(re.split(",", lines[1])[-1]) if tokens[-1] == 'Fail' else -1

                label_test_writer.writerow([str(label), str(cat), str(index), filename.split(".")[0]])

                time_step = 0
                if len(lines) == 129:
                    lines.append(lines[128]+'\n')

                for line in lines[2:129]:
                    line = "".join(line.split())
                    print(time_step, end=" ")

                    line_tokens = re.split("[, \[\]\W]+", line[1:-1])

                    X_test[test_sample, time_step, :] = list(map(int, line_tokens))

                    time_step += 1
                print('\ntest_sample: ', test_sample)
                test_sample += 1
                test_counter += 1
            else:
                print('train')
                print('processing: ' + path + "/" + filename )
                f = open(path + "/" + filename)
                lines = f.readlines()
                index = int(re.split(",", lines[1])[-1]) if tokens[-1] == 'Fail' else -1

                label_train_writer.writerow([str(label), str(cat), str(index), filename.split(".")[0]])

                time_step = 0
                if len(lines) == 129:
                    lines.append(lines[128] + '\n')

                for line in lines[2:129]:
                    line = "".join(line.split())
                    print(time_step, end=" ")

                    line_tokens = re.split("[, \[\]\W]+", line[1:-1])

                    X_train[train_sample, time_step, :] = list(map(int, line_tokens))

                    time_step += 1
                print('\ntrain_sample: ', train_sample)
                train_sample += 1

    label_train.close()
    label_test.close()
    with open("data/Loaded-"+dataset_name+"/data_train.dump", 'wb') as dump_data_file:
        pickle.dump(X_train, dump_data_file, protocol=4)
    with open("data/Loaded-"+dataset_name+"/data_test.dump", 'wb') as dump_data_file:
        pickle.dump(X_test, dump_data_file, protocol=4)


if __name__ == '__main__':
    load("Dataset-12")
