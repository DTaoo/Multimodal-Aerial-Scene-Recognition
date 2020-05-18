
import numpy as np


G_TRAIN_RATIO = 0.8
G_TEST_RATIO = 0.2

def syc_shuffle(arr_list):
    state = np.random.get_state()
    for e in arr_list:
        np.random.shuffle(e)
        np.random.set_state(state)

def split_dataset_ex_with_label_balance_static(view_data,label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0 :
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0 :
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = [i for i in range(num_of_cls)]#random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:int(num_of_cls*TRAIN_RATIO)]]
        test_idx = np.asarray(label_dict[ck])[idx[int(num_of_cls*TRAIN_RATIO):int(num_of_cls*(TRAIN_RATIO+TEST_RATIO))]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)

    #shuffle
    train_shuffle_list = []
    test_shuffle_list = []
    for v in range(nview):
        train_shuffle_list.append(train_view_data[v])
        test_shuffle_list.append(test_view_data[v])

    train_shuffle_list.append(train_label)
    test_shuffle_list.append(test_label)

    syc_shuffle(train_shuffle_list)
    syc_shuffle(test_shuffle_list)

    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)
