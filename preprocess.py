# Functions to load Audioset data from TFRecords and ontology information from .json file
#     - Organizes ontology into dict to obtain top two level ontology classes
#     - Converts to class indices

import os
import json
import csv
import sys

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset


def get_all_children(category, aso):

    childs = aso[category]["child_ids"]
    childs_names = []
    for child in childs:
        child_name = {}
        child_name["name"] = aso[child]["name"]
        if "child_ids" in aso[child]:
            child_name["children"] = get_all_children(child, aso)
        childs_names.append(child_name)
    if childs_names:
        return childs_names


def preprocess_ontology(data_dir):

    f = open(data_dir + 'ontology.json')
    ont_data = json.load(f)
    f.close()

    ont = {}
    for category in ont_data:
        tmp = {}
        tmp["name"] = category["name"]
        tmp["restrictions"] = category["restrictions"]
        tmp["child_ids"] = category["child_ids"]
        tmp["parents_ids"] = []
        ont[category["id"]] = tmp

    for category in ont:  # find parents
        for c in ont[category]["child_ids"]:
            ont[c]["parents_ids"].append(category)

    higher_categories = []  # higher_categories are the ones without parents
    for category in ont:
        if ont[category]["parents_ids"] == []:
            higher_categories.append(category)

    return ont, higher_categories


def get_all_parents(id_, ont, parents_names):

    parents = ont[id_]["parents_ids"]

    if parents:
        for parent in parents:
            parent_name = ont[parents[0]]["name"]
            get_all_parents(parents[0], ont, parents_names)

    parents_names.append(ont[id_]["name"])


def extract_tfrecord_data(data_dir, features_dir):

    with open(data_dir + 'class_labels_indices.csv', mode='r') as file:
        reader = csv.reader(file)
        index_to_id = {rows[0]: rows[1] for rows in reader}

    # List of all .tfrecord files
    tfrecord_files = os.listdir(data_dir + features_dir)

    # Data/class arrays
    sounds_data = []
    class_1 = []
    class_2 = []

    # TFRecord attributes
    context_description = {"video_id": "byte", "labels": "int"}
    sequence_description = {"audio_embedding": "byte"}

    ont, _ = preprocess_ontology(data_dir)
    count = 0

    # Load each tfrecord file
    for filename in tfrecord_files:

        tfrecord_path = data_dir + features_dir + filename
        dataset = TFRecordDataset(tfrecord_path, index_path=None,
                                  description=context_description, sequence_description=sequence_description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for data in iter(loader):
            #data = next(iter(loader))
            data_labels = data[0]['labels'].tolist()[0]
            data_labels = [int(i) for i in data_labels]

            # Collect class labels
            parent_names = []
            for i in data[0]['labels'][0].numpy():
                get_all_parents(index_to_id[str(i)], ont, parent_names)

            file_data = []

            # Collect feature vectors/classes
            if(len(parent_names) > 1):
                for t in data[1]['audio_embedding']:
                    file_data.append(t.numpy())
                class_1.append(parent_names[0])
                class_2.append(parent_names[1])

                sounds_data.append(np.concatenate(file_data))
                count += 1

    print(count)

    sounds_data = np.asanyarray(sounds_data, dtype=object)

    unique_class1 = np.unique(class_1)
    unique_class2 = np.unique(class_2)

    class1_index = np.zeros((len(class_1), )).astype(int)
    for i in range(len(class_1)):
        class1_index[i] = np.where(class_1[i] == unique_class1)[0]

    class2_index = np.zeros((len(class_2), )).astype(int)
    for i in range(len(class_2)):
        class2_index[i] = np.where(class_2[i] == unique_class2)[0]

    return sounds_data, class1_index, class2_index


def class_to_index(data_dir, features_dir):

    with open(data_dir + 'class_labels_indices.csv', mode='r') as file:
        reader = csv.reader(file)
        index_to_id = {rows[0]: rows[1] for rows in reader}

    # List of all .tfrecord files
    tfrecord_files = os.listdir(data_dir + features_dir)

    # Data/class arrays
    class_1 = []
    class_2 = []

    # TFRecord attributes
    context_description = {"video_id": "byte", "labels": "int"}
    sequence_description = {"audio_embedding": "byte"}

    ont, highest_category = preprocess_ontology(data_dir)
    print(highest_category)

    count = 0

    # Load each tfrecord file
    for filename in tfrecord_files:

        tfrecord_path = data_dir + features_dir + filename
        dataset = TFRecordDataset(tfrecord_path, index_path=None,
                                  description=context_description, sequence_description=sequence_description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for data in iter(loader):
            data_labels = data[0]['labels'].tolist()[0]
            data_labels = [int(i) for i in data_labels]

            # Collect class labels
            parent_names = []
            # print(data_labels)
            for i in data[0]['labels'][0].numpy():
                get_all_parents(index_to_id[str(i)], ont, parent_names)
            parent_names = np.unique(parent_names)
            # print(parent_names)

            file_data = []

            # Collect feature vectors/classes
            if(len(parent_names) > 1):
                class_1.append(parent_names[0])
                class_2.append(parent_names[1])

                count += 1

    unique_class1 = np.unique(class_1)
    unique_class2 = np.unique(class_2)

    class_to_index_1 = {unique_class1[i]: i for i in range(len(unique_class1))}
    class_to_index_2 = {unique_class2[i]: i for i in range(len(unique_class2))}

    return class_to_index_1, class_to_index_2, unique_class1, unique_class2


def get_child_index(ont, class_to_index_1, class_to_index_2, unique_class_1, data_dir):

    f = open(data_dir + 'ontology.json')
    ont_data = json.load(f)
    f.close()

    with open(data_dir + 'class_labels_indices.csv', mode='r') as file:
        reader = csv.reader(file)
        class_to_id = {rows[2]: rows[1] for rows in reader}

    ont_matrix = np.zeros((len(class_to_index_1), len(class_to_index_2)))
    for class_ in unique_class_1:
        for cat in ont_data:
            if cat["name"] == class_:
                id_ = cat["id"]

        children = ont[id_]['child_ids']
        for child in children:
            class_2_name = ont[child]['name']
            ont_matrix[class_to_index_1[class_]
                       ][class_to_index_2[class_2_name]] = 1

        # print(children)


# Main
if __name__ == "__main__":

    data_dir = './'
    train_features_dir = 'audioset_v1_embeddings/bal_train/'

    #train_data, class1_index, class2_index = extract_tfrecord_data(data_dir, train_features_dir)

#     # Save all train data files
#     save_dir = sys.argv[1]
#     np.save(save_dir + 'audioset_data.npy', train_data)
#     np.save(save_dir + 'audioset_labels_1.npy', class1_index)
#     np.save(save_dir + 'audioset_labels_2.npy', class2_index)

    class_to_index_1, class_to_index_2, unique_class_1, unique_class_2 = class_to_index(
        data_dir, train_features_dir)
    # print(class_to_index_1)
    print(class_to_index_2)

    ont, _ = preprocess_ontology(data_dir)
    # print(ont)
    get_child_index(ont, class_to_index_1, class_to_index_2,
                    unique_class_1, data_dir)

    #### Get val set from small subset of unbalanced train ####
#     val_features_dir = 'audioset_v1_embeddings/unbal_train/'

#     all_val_data, class1_index, class2_index = extract_tfrecord_data(data_dir, val_features_dir)

#     num_val = 0.15 * len(train_data)
#     num_class_2 = len(np.unique(class2_index))
#     num_val_per_class = int(num_val / num_class_2)

#     val_data = []
#     val_class1_index = np.zeros((num_val_per_class*num_class_2, )).astype(int)
#     val_class2_index = np.zeros((num_val_per_class*num_class_2, )).astype(int)
#     np.random.seed(0)
#     for idx in range(num_class_2):
#         idx_match = np.where(class2_index == idx)[0]
#         val_idx = np.random.choice(idx_match, num_val_per_class)

#         for i in range(num_val_per_class):
#             val_data.append(all_val_data[val_idx[i]])
#             val_class1_index[idx * num_class_2 + i] = class1_index[val_idx[i]]
#             val_class2_index[idx * num_class_2 + i] = idx


#     # Save all data files
#     save_dir = sys.argv[1]
#     np.save(save_dir + 'audioset_val_data.npy', np.asanyarray(val_data, dtype=object))
#     np.save(save_dir + 'audioset_val_labels_1.npy', val_class1_index)
#     np.save(save_dir + 'audioset_val_labels_2.npy', val_class2_index)
