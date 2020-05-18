from torch.utils.data import Dataset
import json
import cv2
import numpy as np

import time

class CVSDataset(Dataset):

    def __init__(self,root_dir,split = 'train'): # 'train' 'val' 'test'
        self.root_dir = root_dir
        self.split = split

        # load split
        self.json_dict = dict(json.load(open(root_dir+'/split/'+split+'.json')))
        self.image_file_list = self.json_dict['image_file_list']
        self.audio_file_list = self.json_dict['audio_file_list']
        self.label_list = np.asarray(self.json_dict['label']).astype(int)

        self.length = len(self.label_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # return (img_data,audio_data,label)

        image_file = self.root_dir+self.image_file_list[item]
        image_data = np.transpose(cv2.imread(image_file),[2,0,1]) # Channel


        audio_file = self.root_dir+self.audio_file_list[item]
        audio_data = np.load(audio_file)

        label = self.label_list[item]

        return np.asarray(image_data).astype(float),np.asarray(audio_data).astype(float),label


if __name__ == '__main__':
    #a = librosa.load()


    dataset = CVSDataset('F:/download/CVS_Dataset_New/',split='train')

    for i in range(len(dataset)):
        print(i)
        time_begin = time.time()
        img,aud,y = dataset[i] # 加载一个的时间 平均一个样本0.08秒 ，所以一个batch 16 数据加载约等于 1s 可以接受。如果dataloader 可以多线程
        time_end = time.time()
        print(time_end-time_begin)
        print(img.shape)
        print(aud.shape)
        print(y)






