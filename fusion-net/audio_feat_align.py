# 部分声音由于过短，导致无法批量训练，因此进行padding 补0 使得shape一致
import  os
import numpy as np

AUDIO_FEAT_PATH = 'F:/download/CVS_Dataset_New/audio_feat/'


files = os.listdir(AUDIO_FEAT_PATH)

count = 0

for i,f in enumerate(files):
    print("processing:%d/%d" %(i,len(files)))
    feat = np.load(AUDIO_FEAT_PATH+f)
    # shape[1] == 400
    if feat.shape[1] != 400:
        new_feat = np.zeros((1,400,64))
        new_feat[0,0:feat.shape[1],0:64] = feat
        # remov
        os.remove(AUDIO_FEAT_PATH+f)
        print("find :%s" % f)
        count+=1
        np.save(AUDIO_FEAT_PATH+f,new_feat)
        #print(new_feat.shape)

print("processed %d files" % count)