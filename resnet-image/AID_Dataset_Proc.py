import argparse
import os
import scipy.io as scio
from PIL import Image
import numpy as np
import utils
import pickle




def main():
    parser = argparse.ArgumentParser(description='AID_Dataset_Proc')
    parser.add_argument('--aid_path',type=str,default='/mnt/scratch/hudi/soundscape/data/AID/',help=' the path of aid dataset')
    parser.add_argument('--crop_size',type=int,default=512,help='crop size')
    parser.add_argument('--save_file',type=str,default='proc_aid.pkl',help='saved .mat file name')

    args = parser.parse_args()

    aid_path = args.aid_path
    crop_size = args.crop_size

    # 列举类别
    class_names = os.listdir(aid_path)
    class_name2index = dict()
    for i,e in enumerate(class_names):
        class_name2index[e] = i
    # 列举所有图片
    num_of_class = len(class_name2index.keys())
    image_data = []
    image_label = []

    for cn in class_names[:]:
        cn_path = os.path.join(args.aid_path, cn)
        imgs = os.listdir(cn_path)
        for img_file in imgs:
            img = Image.open(os.path.join(cn_path, img_file)).convert('RGB').resize((crop_size,crop_size))
            print(img_file)
            rs_img = np.transpose(img,[2,0,1])
            image_data.append(rs_img)
            image_label.append(class_name2index[cn])

    # save
    image_data = np.asarray(image_data).astype('uint8')
    image_label = np.asarray(image_label)
    save_name = args.save_file

    data_dict =dict()

    train_X, train_Y, test_X, test_Y = utils.split_dataset_ex_with_label_balance_static([image_data],image_label,train_ratio=0.8,test_ratio=0.2)


    data_dict['tr_X'] = train_X[0]
    data_dict['tr_Y'] = train_Y
    data_dict['te_X'] = test_X[0]
    data_dict['te_Y'] = test_Y

    #np.save('./data/'+save_name,data_dict)
    pickle.dump(data_dict,open('../data/'+save_name,mode='wb'),protocol=4)
    #scio.savemat('./data/'+save_name,data_dict)



if __name__ == '__main__':
    main()