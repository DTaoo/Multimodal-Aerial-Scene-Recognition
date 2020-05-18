'''
将数据集整理到一个文件夹内
datasetDir\ # 根目录
    \image\ # 图像数据 *.jpg
    \audio\ # 声音数据 *.wav
    \split\ # 数据划分 train.json val.json test.json

'''
import os
import numpy as np
import cv2
import shutil

# 每个样本都有一个ID，一个ID对应一个图片和音频
# 先提取出有效的ID

ID_LIST_FILE = '../data/file_name.txt' # id<->audio
DATASET_DIR = 'F:\\download\\Dataset_v2\\' # 高质量图片数据集目录
CVS_DATSET_DIR = 'F:\\download\\CVS_Dataset\\' # 原图像+声音数据集
DST_DATASET_DIR = 'F:\\download\\CVS_Dataset_New\\'

IMAGE_CROP_SIZE = 512

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# 取出所有id与声音 pair
'''
00000_32.7306856627_35.071105957.mp3
00001_32.0544628275184_34.7529530525208.mp3
00002_32.054462827_34.752953052.mp3
'''
id_list = [e.strip().split('_') for e in open(ID_LIST_FILE).readlines()] # id,x,y
for i,e in enumerate(id_list): # remove .mp3
    id_list[i][2] = e[2].replace('.mp3','')

print(id_list[0])

# x+y->id
# 不同ID可能会有相同的经纬度！！！！！！！！
# 00023_32.6635119579096_-114.603710174561.mp3
# 00024_32.6635119579096_-114.603710174561.mp3
xy_mapping_dict = dict()
for e in id_list:
    xy_mapping_dict[e[0]] = e[1]+e[2]


# label 个数
label_list = os.listdir(DATASET_DIR+'Dataset') #磨出最后一个类
label_list.remove('wind turbine')
label_id = [e for e in range(len(label_list))]
label_dict = dict()

for i,l in enumerate(label_list):
    label_dict[l] = i

# 开始处理

# 读取所有数据
# 读取图像数据

image_file_list = [] # 所有图像的路径 相对于 DATSET_DIR
image_label_list = [] # 所有图像的LABEL
valid_id_list = []
valid_image_id = dict() # 有label样本的id
for l in label_list:
    imgs = os.listdir(DATASET_DIR+'dataset/'+l+'/')
    for img in imgs:
        image_file_list.append('dataset/'+l+'/'+img)
        image_label_list.append(label_dict[l])
        id = img.replace('.jpg','')

        valid_image_id[id] = True
        valid_id_list.append(id)

# 读取音频
audio_file_dict = dict() # xy->audio_file 相对于CVS_DATASET_DIR

audio_root_path = CVS_DATSET_DIR +'sound'
audio_dirs = os.listdir(audio_root_path)

for dir in audio_dirs:
    audio_dirs_2 = os.listdir(audio_root_path+'/'+ dir)
    for dir2 in audio_dirs_2:
        audios = os.listdir(audio_root_path+'/'+dir+'/'+dir2)
        for audio in audios:
            a = audio.replace('.mp3','')
            xy = a.split('_')[-2:]
            xy_cat = xy[0]+xy[1]
            audio_file_dict[xy_cat] = 'sound'+'/' + dir + '/' + dir2 + '/' + audio

# 对齐
audio_file_list = []
for id in valid_id_list:
    if '_' in id:
        clear_id = id.split('_')[0]
    else:
        clear_id = id
    xy = xy_mapping_dict[clear_id]
    audio_file_list.append(audio_file_dict[xy])

print(len(image_file_list))
print(len(audio_file_list))

assert (len(image_file_list) == len(image_file_list))

# 将图像复制到新目录下


new_image_file_list = [] # 相对于 DST_DATASET_DIR
new_audio_file_list = [] # 相对于 DST_DATASET_DIR
new_image_label_lsit = image_label_list # 不变

num = len(image_file_list)

B_PROCESS_IMAGE = False # 如果想变换划分 改为False
B_PROCESS_AUDIO = False

for i in range(num):
    print("processing:%d/%d" % (i+1,num))
        # 保持512
    path = 'image/' + valid_id_list[i] + '.jpg'
    new_image_file_list.append(path)
    if B_PROCESS_IMAGE:
        #img = cv2.imread(DATASET_DIR+'\\'+image_file_list[i])
        org_path = DATASET_DIR+'/'+image_file_list[i]
        #img = cv2.resize(img,(IMAGE_CROP_SIZE,IMAGE_CROP_SIZE)).astype('uint8')
        dst_path =  DST_DATASET_DIR + path
        shutil.copy(org_path, dst_path)
        #cv2.imwrite(DST_DATASET_DIR+path,img)

    path = 'audio_feat/' + valid_id_list[i] + '.npy'
    new_audio_file_list.append(path)
    if B_PROCESS_AUDIO:
        # audio
        org_path = CVS_DATSET_DIR + audio_file_list[i]
        dst_path = DST_DATASET_DIR + 'audio/' + valid_id_list[i] + '.mp3'
        if not os.path.exists(dst_path):
            shutil.copy(org_path,dst_path)


# split

new_image_file_list = np.asarray(new_image_file_list)
new_audio_file_list = np.asarray(new_audio_file_list)
new_image_label_lsit = np.asarray(new_image_label_lsit)

import random

train_num = int(len(new_image_label_lsit)*TRAIN_RATIO)
val_num = int(len(new_image_label_lsit)*VAL_RATIO)
test_num = int(len(new_image_label_lsit)*TEST_RATIO)

random_idx = random.sample(range(len(new_image_label_lsit)),k=len(new_image_label_lsit))

train_idx = random_idx[:train_num]
val_idx = random_idx[train_num:val_num+train_num]
test_idx = random_idx[train_num+val_num:train_num+val_num+test_num]

train_split = dict()

train_split['image_file_list'] = list(new_image_file_list[train_idx])
train_split['audio_file_list'] = list(new_audio_file_list[train_idx])
train_split['label'] = list(new_image_label_lsit[train_idx].astype('float'))

val_split = dict()
val_split['image_file_list'] = list(new_image_file_list[val_idx])
val_split['audio_file_list'] = list(new_audio_file_list[val_idx])
val_split['label'] = list(new_image_label_lsit[val_idx].astype('float'))

test_split = dict()
test_split['image_file_list'] = list(new_image_file_list[test_idx])
test_split['audio_file_list'] = list(new_audio_file_list[test_idx])
test_split['label'] = list(new_image_label_lsit[test_idx].astype('float')) # 如果int json无法序列化 不知道为什么 = =

import json

json.dump(train_split,open(DST_DATASET_DIR+'split/train.json',mode='w'))
json.dump(val_split,open(DST_DATASET_DIR+'split/val.json',mode='w'))
json.dump(test_split,open(DST_DATASET_DIR+'split/test.json',mode='w'))

print('finished.')








