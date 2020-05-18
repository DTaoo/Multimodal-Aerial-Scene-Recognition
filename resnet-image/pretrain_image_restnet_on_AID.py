import argparse
import os
import scipy.io as scio
import cv2 as cv
import numpy as np
import utils
import resnet_image
import pickle
import torch.optim as optim
import torchvision as tv
import torch as t
import datetime
import sys
from torch.utils.data import DataLoader,TensorDataset



def net_train(net,data_loader,opt,loss_func,cur_e,args):
    # 父类方法
    net.train()
    begin_time = datetime.datetime.now()

    train_loss = 0.0
    batch_num = int(len(data_loader.dataset) / args.batch_size)

    for i,data in enumerate(data_loader,0):
        #print('batch:%d/%d' % (i,batch_num))
        img,label = data
        img,label = img.cuda(),label.cuda()
        # 归零梯度
        opt.zero_grad()
        # 输出
        output = net(img)[1]
        loss = loss_func(output, label)
        loss.backward() # 反向
        opt.step()

        # loss
        train_loss += loss

    end_time = datetime.datetime.now()
    delta_time = (end_time-begin_time)
    delta_seconds = (delta_time.seconds*1000 + delta_time.microseconds)/1000

    print('epoch:%d loss:%.4f time:%.4f'% (cur_e,train_loss.cpu(),(delta_seconds)))
    return net


def net_test(net,data_loader):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    with t.no_grad():
        for i, data in enumerate(data_loader, 0):

            img, label = data # cpu
            img, label = img.cuda(), label.cuda() # gpu
            output = net(img)[1]

            predict_label = t.argmax(output,dim=1)
            correct += (predict_label == label).sum()

    return correct.cpu().numpy()/num




def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset', type=str, default='../data/proc_aid.pkl', help='the path of aid dataset')
    parser.add_argument('--batch_size', type=int, default=32,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='training batch size')
    parser.add_argument('--epoch',type=int,default=50,help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='0', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    args = parser.parse_args()
    # 加载数据
    print('loading data from:'+args.dataset)
    data_dict = pickle.load(open(args.dataset,mode='rb'))
    print('data loaded.')

    train_dataset = TensorDataset(t.FloatTensor(data_dict['tr_X']),t.LongTensor(data_dict['tr_Y'])) # 不需要变成one hot
    test_dataset = TensorDataset(t.FloatTensor(data_dict['te_X']),t.LongTensor(data_dict['te_Y']))

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)


    loss_func = t.nn.CrossEntropyLoss()


    # 初始化模型
    net = resnet_image.resnet101(False,num_classes= 30)
    # 将模型转为cuda类型 适应多GPU并行训练
    gpu_ids = [int(e) for e in args.gpu_ids.split(',')]
    net = t.nn.DataParallel(net, device_ids=[0,1,2,3]).cuda()

    # 优化器
    optimizer = optim.SGD(params=net.parameters(),lr=args.learning_rate, momentum=args.momentum)
    #optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0001)
    max_acc = 0.0
    for e in range(args.epoch):

        net.train()
        begin_time = datetime.datetime.now()

        train_loss = 0.0
        batch_num = int(len(train_dataloader.dataset) / args.batch_size)

        for i, data in enumerate(train_dataloader, 0):
            # print('batch:%d/%d' % (i,batch_num))
            img, label = data
            img, label = img.cuda(), label.cuda()
            # 归零梯度
            optimizer.zero_grad()
            # 输出
            output = net(img)[1]
            loss = loss_func(output, label)
            loss.backward()  # 反向
            optimizer.step()

            # loss
            train_loss += loss
        end_time = datetime.datetime.now()
        delta_time = (end_time - begin_time)
        delta_seconds = (delta_time.seconds * 1000 + delta_time.microseconds) / 1000

        test_acc = net_test(net,test_dataloader)
        print('epoch:%d loss:%.4f time:%.4f test acc:%f' % (e, train_loss.cpu(), (delta_seconds), test_acc))
        sys.stdout.flush()
        if test_acc > max_acc:
            max_acc = test_acc
            try:
                state_dict = net.module.state_dict()
            except AttributeError:
                state_dict = net.state_dict()
            t.save(state_dict, '../model/visual_model_pretrain.pt')


if __name__ == '__main__':
    main()


