import sys

sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("../data/")
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
from CVS_dataset import CVSDataset
from resnet_image import resnet101 as IMG_NET
from resnet_audio import resnet50  as AUD_NET
from fusion_net import FusionNet as FUS_NET
from fusion_net import FusionNet_KD
import pickle
import torch.optim as optim
import datetime
import os
import sklearn
from data_partition import data_construction
import numpy as np
import cv2
import os

global features_blobs

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    weight_softmax = weight_softmax[:,0:2048]

    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset_dir', type=str, default='F:\\download\\CVS_Dataset_New\\',
                        help='the path of the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--av_net_ckpt', type=str, default='kd_checkpoint49_0.750.pt', help='av net checkpoint')

    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/soundscape/data/',
                        help='image net weights')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--data_name', type=str, default='CVS_data_ind.pkl')
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()

    class_name = ['forest', 'harbour', 'farmland', 'grassland', 'airport', 'sports land', 'bridge',
                  'beach', 'residential', 'orchard', 'train station', 'lake', 'sparse shrub land']

    print('kd_model...')

    global features_blobs

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_construction(args.data_dir)

    test_dataset = CVSDataset(args.data_dir, test_sample, test_label, seed=args.seed, enhance=False,
                              event_label_name='event_label_bayes_59')

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_threads)


    image_net = IMG_NET(num_classes=30)
    audio_net = AUD_NET()
    fusion_net = FusionNet_KD(image_net, audio_net, num_classes=13)

    fusion_net = torch.nn.DataParallel(fusion_net, device_ids=[0]).cuda()

    MODEL_PATH = 'checkpoint2'
    MODEL_FILE = os.path.join(MODEL_PATH, args.av_net_ckpt)
    state = torch.load(MODEL_FILE)
    fusion_net.load_state_dict(state['model'])

    fusion_net.eval()

    # get the softmax weight
    params = list(fusion_net.parameters())
    weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

    fusion_net._modules.get('module')._modules.get('image_net')._modules.get('layer4').register_forward_hook(hook_feature)

    with torch.no_grad():
        count=0
        for i, data in enumerate(test_dataloader, 0):
            img, aud, label, _e, _r = data
            img, aud, label = img.type(torch.FloatTensor).cuda(), aud.type(torch.FloatTensor).cuda(), label.type(torch.LongTensor).cuda() # gpu
            logit, _ = fusion_net(img, aud)

            h_x = F.softmax(logit, dim=1).data.squeeze()
            for j in range(logit.shape[0]):
                h_x_current = h_x[j,:]
                probs, idx = h_x_current.sort(0, True)
                probs = probs.cpu().numpy()
                idx = idx.cpu().numpy()

                CAMs = returnCAM(features_blobs[0][j], weight_softmax, [label[j]])

                current_img = np.transpose(img[j].cpu().numpy(),[1,2,0])
                height, width, _ = current_img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.3 + current_img * 0.5
                
                if not os.path.exists('cam3/kd/' + args.av_net_ckpt): 
                    os.mkdir('cam3/kd/' + args.av_net_ckpt) 
                #cv2.imwrite(os.path.join('cam/result/kd/', args.av_net_ckpt, class_name[label[j]]+'_%04d.jpg' % j), result)
                file_name = '%04d_'%count + class_name[label[j]] + '_' + class_name[idx[0]] + '_%.3f' % h_x_current[label[j]] + '.jpg'
                cv2.imwrite(os.path.join('cam3/kd/', args.av_net_ckpt, file_name), result)
                count += 1
            features_blobs = []


if __name__ == '__main__':
    main()






