import sys
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("../data/")
import torch
from torch.utils.data import DataLoader,TensorDataset
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

def net_train(net,data_loader,opt,loss_func,cur_e,args):
    net.train()
    begin_time = datetime.datetime.now()

    train_loss = 0.0
    batch_num = int(len(data_loader.dataset) / args.batch_size)

    for i,data in enumerate(data_loader, 0):
        #print('batch:%d/%d' % (i,batch_num))
        img, aud, label = data
        img, aud, label = img.type(torch.FloatTensor).cuda(),aud.type(torch.FloatTensor).cuda(),label.type(torch.LongTensor).cuda()

        opt.zero_grad()

        output = net(img, aud)
        loss = loss_func(output, label)
        loss.backward()
        opt.step()

        train_loss += loss.cpu()

    end_time = datetime.datetime.now()
    delta_time = (end_time-begin_time)
    delta_seconds = (delta_time.seconds*1000 + delta_time.microseconds)/1000

    print('epoch:%d loss:%.4f time:%.4f'% (cur_e,train_loss.cpu(),(delta_seconds)))


def net_test(net,data_loader,cal_map=False):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    predict_labels = np.array([])
    ground_labels  = np.array([])
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            img, aud, label = data
            img, aud, label = img.type(torch.FloatTensor).cuda(), aud.type(torch.FloatTensor).cuda(), label.type(torch.LongTensor).cuda() # gpu
            output = net(img, aud)
            predict_label = torch.argmax(output, dim=1)
            #if predict_labels == None:
            #    predict_labels = predict_label.cpu().numpy()
            #else:
            predict_labels = np.concatenate((predict_labels, predict_label.cpu().numpy()))
            #if ground_labels == None:
            #    ground_labels = label.cpu().numpy()
            #else:
            ground_labels = np.concatenate((ground_labels, label.cpu().numpy()))
            correct += ((predict_label == label).sum().cpu().numpy())
            # map

    
    (precision, recall, fscore, sup) = sklearn.metrics.precision_recall_fscore_support(ground_labels, predict_labels, average='weighted')
    acc = correct/num

    return (acc, precision, recall, fscore)


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor


def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset_dir', type=str, default='F:\\download\\CVS_Dataset_New\\', help='the path of the dataset')
    parser.add_argument('--batch_size', type=int, default=64,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='training batch size')
    parser.add_argument('--epoch',type=int,default=2000,help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    parser.add_argument('--image_net_weights', type=str, default='visual_model_pretrain.pt', help='image net weights')
    parser.add_argument('--audio_net_weights', type=str, default='audio_pretrain_net.pt',
                        help='image net weights')

    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/soundscape/data/',
                        help='image net weights')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--data_name', type=str, default='CVS_data_ind.pkl')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--audionet_pretrain', type=int, default=1)
    parser.add_argument('--videonet_pretrain', type=int, default=1)

    args = parser.parse_args()

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_construction(args.data_dir)

    #f = open(args.data_name, 'wb')
    #data = {'train_sample':train_sample, 'train_label':train_label, 'test_sample':test_sample, 'test_label':test_label}
    #pickle.dump(data, f)
    #f.close()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    train_dataset = CVSDataset(args.data_dir, train_sample, train_label, seed=args.seed)
    val_dataset  = CVSDataset(args.data_dir, val_sample, val_label, seed=args.seed)
    test_dataset = CVSDataset(args.data_dir, test_sample, test_label, seed=args.seed)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_threads)
    val_dataloader   = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads)
    test_dataloader  = DataLoader(dataset=test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_threads)

    image_net = IMG_NET(num_classes=30)
    if args.videonet_pretrain:
        state = torch.load(args.image_net_weights)
        image_net.load_state_dict(state)

    audio_net = AUD_NET()
    if args.audionet_pretrain:
        state = torch.load(args.audio_net_weights)['model']
        audio_net.load_state_dict(state)

    # all stand up
    fusion_net = FUS_NET(image_net, audio_net, num_classes=13)
    

    gpu_ids = [i for i in range(4)]
    fusion_net_cuda = torch.nn.DataParallel(fusion_net, device_ids=gpu_ids).cuda()

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=fusion_net_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0001)

    max_fscore = 0.

    for e in range(args.epoch):

        fusion_net_cuda.train()
        begin_time = datetime.datetime.now()

        train_loss = 0.0
        batch_num = int(len(train_dataloader.dataset) / args.batch_size)

        for i, data in enumerate(train_dataloader, 0):
            # print('batch:%d/%d' % (i,batch_num))
            img, aud, label = data
            img, aud, label = img.type(torch.FloatTensor).cuda(), aud.type(torch.FloatTensor).cuda(), label.type(
                torch.LongTensor).cuda()

            optimizer.zero_grad()

            output = fusion_net_cuda(img, aud)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu()

        end_time = datetime.datetime.now()
        delta_time = (end_time - begin_time)
        delta_seconds = (delta_time.seconds * 1000 + delta_time.microseconds) / 1000

        (val_acc, val_precision, val_recall, val_fscore) = net_test(fusion_net_cuda, val_dataloader)
        print('epoch:%d loss:%.4f time:%.4f val acc:%.4f val_precision:%.4f val_recall:%.4f val_fscore:%.4f ' % (e, train_loss.cpu(), (delta_seconds), val_acc, val_precision, val_recall, val_fscore))
        if val_fscore > max_fscore:
            max_fscore = val_fscore
            (test_acc, test_precision, test_recall, test_fscore) = net_test(fusion_net_cuda, test_dataloader)
            print('test acc:%.4f precision:%.4f recall:%.4f fscore:%.4f' % (test_acc, test_precision, test_recall, test_fscore))

        if e in [30, 60, 90]:
            decrease_learning_rate(optimizer, 0.1)
            print('decreased learning rate by 0.1')


if __name__ == '__main__':
    main()






