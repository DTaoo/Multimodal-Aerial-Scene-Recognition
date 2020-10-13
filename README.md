# Multimodal-Aerial-Scene-Recognition
This is the code for paper [Cross-Task Transfer for Geotagged Audiovisual Aerial Scene Recognition](https://arxiv.org/abs/2005.08449) (ECCV 2020)

## Pretrained weight
The pretrain weights of audio and visual net, downloaded from here:
https://pan.baidu.com/s/106Y0H5xTXYwi4Zeq2Rw_Og  code:5es6
or here:
https://zenodo.org/record/4082894

## Dataset
We construct a new dataset, named AuDio Visual Aerial sceNe reCognition datasEt (ADVANCE), providing 5075 paired images and sound clips categorized to 13 scenes, for exploring the aerial scene recognition task. You can view the dataset [here](https://akchen.github.io/ADVANCE-DATASET/) or directly
download the dataset [here](https://zenodo.org/record/3828124). Related train/val/test partition can be achieved by the 'data_construction()' function in data/data_partition.py

## Usage
The three branchs of cross-modal transfer methods are in the model/ folder, i.g, sq_transfer.py, kl_transfer.py, and bayes_transfer.py. Please extract the sound event knwoledge/prediction using audio_event_extactor.py firstly, then run kl_transfer.py or bayes_transfer.py.


## Reference
If you use this repo or the ADVANCE dataset in your research, please cite our paper:

    Cross-Task Transfer for Geotagged Audiovisual Aerial Scene Recognition 
    Di Hu, Xuhong Li, Lichao Mou, Pu Jin, Dong Chen, Liping Jing, Xiaoxiang Zhu, Dejing Dou
    ECCV 2020
