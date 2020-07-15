import sys
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("../data/")
import torch
from torch.utils.data import DataLoader,TensorDataset
import argparse
from CVS_dataset import CVS_Audio
from resnet_audio import resnet50  as AUD_NET
import pickle
import torch.optim as optim
import datetime
import os
import sklearn
from data_partition import audio_construction
import numpy as np

def net_test(net,data_loader):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    predict_labels = np.array([])
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            aud, label = data
            aud = aud.type(torch.FloatTensor).cuda()
            output = net(aud)[1]        # here for extracting the event predictions
            output_cpu = output.cpu().numpy()
            save_sound_event_label(label, output_cpu)
            
            #if predict_labels.shape[0] == 0: 
            #    predict_labels = output_cpu
            #else:
            #    predict_labels = np.concatenate((predict_labels, output_cpu), axis=0)

    #return predict_labels

def save_sound_event_label(data_sample, data_label, data_dir='/mnt/scratch/hudi/soundscape/data/'):

    for i in range(len(data_sample)):
        data_m = data_sample[i].split('/')
        data_path = os.path.join(data_dir, 'event_label_bayes', data_m[0])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_name = data_m[1]+'_ser.npy'
        np.save(os.path.join(data_path, data_name), data_label[i])
 
           

def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset_dir', type=str, default='F:\\download\\CVS_Dataset_New\\', help='the path of the dataset')
    parser.add_argument('--batch_size', type=int, default=64,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='training batch size')
    parser.add_argument('--epoch',type=int,default=2000,help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    parser.add_argument('--image_net_weights', type=str, default='visual_model_pretrain.pt', help='image net weights')
    parser.add_argument('--audio_net_weights', type=str, default='checkpoint59.pt',
                        help='image net weights')

    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/soundscape/data/',
                        help='image net weights')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--data_name', type=str, default='CVS_data_ind.pkl')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--audionet_pretrain', type=int, default=1)
    parser.add_argument('--videonet_pretrain', type=int, default=1)

    args = parser.parse_args()

    data_sample = audio_construction(args.data_dir)

    audio_dataset = CVS_Audio(args.data_dir, data_sample)
    
    audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads)
   
    audio_net = AUD_NET()
    state = torch.load(args.audio_net_weights)['model']
    audio_net.load_state_dict(state)

    audio_net = audio_net.cuda()
    predict_events = net_test(audio_net, audio_dataloader)
    #save_sound_event_label(data_sample, predict_events, args.data_dir)

if __name__ == '__main__':
    main()






