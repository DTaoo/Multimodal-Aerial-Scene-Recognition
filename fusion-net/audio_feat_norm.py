# 音频特征 归一化

import os
import numpy as np
AUDIO_FEAT_PATH = 'F:/download/CVS_Dataset_New/audio_feat/'

feat_files = os.listdir(AUDIO_FEAT_PATH)

data_list = []

for i,f in enumerate(feat_files):
    print('loading :%d/%d' %(i,len(feat_files)))
    data = np.load(AUDIO_FEAT_PATH + f)
    data_list.append(data)

mu = np.mean(data_list)
sigma = np.std(data_list)

print(mu)
print(sigma)

for i,f in enumerate(feat_files):
    print('saving :%d/%d' % (i,len(feat_files)))
    new_data = (data_list[i] - mu)/sigma
    np.save(AUDIO_FEAT_PATH+feat_files[i],new_data)