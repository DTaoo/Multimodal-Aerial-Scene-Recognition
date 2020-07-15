import os
import numpy as np


def get_dir_label(data_dir):
    cate_list = os.listdir(os.path.join(data_dir, 'Dataset_v3_vision'))
    if '.DS_Store' in cate_list:
        cate_list.remove('.DS_Store')
    class_id_dict = {}
    for i in range(len(cate_list)):
        current_class = cate_list[i]
        class_dir = os.path.join(data_dir, 'Dataset_v3_vision', current_class)
        sample_list = os.listdir(class_dir)
        if '.DS_Store' in sample_list:
            sample_list.remove('.DS_Store')
        data_samples = []
        for sample in sample_list:
            data_samples.append(os.path.join(current_class, sample[:-4]))
        class_id_dict[i] = data_samples
    return class_id_dict

def integrate_sample_id(sample_list):
    sample_id_dict = {}
    for i in range(len(sample_list)):
        sample_ = sample_list[i].split('/')
        sample_id = sample_[1][:5]
        if sample_id not in sample_id_dict:
             sample_id_dict[sample_id] = []
        sample_id_dict[sample_id].append(sample_list[i])
    return sample_id_dict

def get_samples(sample_id_dict, samples):
    sample_list = []
    for i in range(len(samples)):
        sample_list.extend(sample_id_dict[samples[i]])

    return sample_list

def data_splitter(class_id_dict, train_ratio, val_ratio):
    id_list = list(class_id_dict.keys())
    train_sample = []
    train_label  = []
    test_sample  = []
    test_label   = []
    val_sample   = []
    val_label    = []

    for i in range(len(id_list)):
        samples = class_id_dict[i]
        np.random.seed(i)
        np.random.shuffle(samples)
        train_num = int(np.floor(len(samples)*train_ratio))
        val_num   = int(np.floor(len(samples)*val_ratio))
        train_sample.extend(samples[:train_num])
        train_label.extend([i for k in range(train_num)])

        val_sample.extend(samples[train_num:(train_num+val_num)])
        val_label.extend([i for k in range(val_num)])

        test_sample.extend(samples[(train_num+val_num):])
        test_label.extend([i for k in range(len(samples)-train_num-val_num)])

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)

def data_splitter_by_id(class_id_dict, train_ratio, val_ratio):
    id_list = list(class_id_dict.keys())
    train_sample = []
    train_label  = []
    test_sample  = []
    test_label   = []
    val_sample   = []
    val_label    = []

    for i in range(len(id_list)):
        samples = class_id_dict[i]
        sample_id_dict = integrate_sample_id(samples)
        sample_ids = list(sample_id_dict.keys())

        np.random.seed(i)
        np.random.shuffle(sample_ids)

        train_num = int(np.floor(len(sample_ids)*train_ratio))
        val_num   = int(np.floor(len(sample_ids)*val_ratio))
        current_train_sample_id = sample_ids[:train_num]
        current_val_sample_id   = sample_ids[train_num:(train_num+val_num)]
        current_test_sample_id  = sample_ids[(train_num+val_num):]

        current_train_sample = get_samples(sample_id_dict, current_train_sample_id)
        train_sample.extend(current_train_sample)
        train_label.extend([i for k in range(len(current_train_sample))])

        current_val_sample = get_samples(sample_id_dict, current_val_sample_id)
        val_sample.extend(current_val_sample)
        val_label.extend([i for k in range(len(current_val_sample))])

        current_test_sample = get_samples(sample_id_dict, current_test_sample_id)
        test_sample.extend(current_test_sample)
        test_label.extend([i for k in range(len(current_test_sample))])

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)



def cate_data_sample(class_id_dict, train_ratio, class_id):
    id_list = list(class_id_dict.keys())
    
    samples = class_id_dict[class_id]
    np.random.seed(class_id)
    np.random.shuffle(samples)
    train_num = int(np.floor(len(samples)*train_ratio))
    train_sample = samples[:train_num]
 
    return train_sample

def data_construction(data_dir, train_ratio=0.7, val_ratio=0.1):

    class_id_dict = get_dir_label(data_dir)

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_splitter_by_id(class_id_dict, train_ratio, val_ratio)

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)


def audio_construction(data_dir, train_ratio=0.7, val_ratio=0.1):

    class_id_dict = get_dir_label(data_dir)

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_splitter_by_id(class_id_dict, train_ratio, val_ratio)

    train_sample.extend(val_sample)
    train_sample.extend(test_sample)

    return train_sample


def single_category_construction(data_dir, train_ratio=0.7, class_id=0):

    class_id_dict = get_dir_label(data_dir)

    data_sample = cate_data_sample(class_id_dict, train_ratio, class_id)

    return data_sample


