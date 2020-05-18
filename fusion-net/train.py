import sys
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
import torch as t
from torch.utils.data import DataLoader,TensorDataset
import argparse
from CVS_Dataset import CVSDataset
from resnet_image import resnet101 as IMG_NET
from resnet_audio import resnet50  as AUD_NET
from fusion_net import FusionNet as FUS_NET
import pickle
import torch.optim as optim
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def net_train(net,data_loader,opt,loss_func,cur_e,args):
    # 父类方法
    net.train()
    begin_time = datetime.datetime.now()

    train_loss = 0.0
    batch_num = int(len(data_loader.dataset) / args.batch_size)

    for i,data in enumerate(data_loader,0):
        print('batch:%d/%d' % (i,batch_num))
        img,aud,label = data
        img,aud,label = img.type(t.FloatTensor).cuda(),aud.type(t.FloatTensor).cuda(),label.type(t.LongTensor).cuda()
        # 归零梯度
        opt.zero_grad()
        # 输出
        output = net(img,aud)
        loss = loss_func(output,label)
        loss.backward() # 反向
        opt.step()

        # loss
        train_loss += loss.cpu()


    end_time = datetime.datetime.now()
    delta_time = (end_time-begin_time)
    delta_seconds = (delta_time.seconds*1000 + delta_time.microseconds)/1000

    print('epoch:%d loss:%.4f time:%.4f'% (cur_e,train_loss.cpu(),(delta_seconds)))


def net_test(net,data_loader):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    with t.no_grad():
        for i, data in enumerate(data_loader, 0):

            img,aud,label = data # cpu
            img,aud,label = img.type(t.FloatTensor).cuda(),aud.type(t.FloatTensor).cuda(),label.type(t.LongTensor).cuda() # gpu
            output = net(img,aud)

            predict_label = t.argmax(output,dim=1)
            correct += ((predict_label == label).sum().cpu().numpy())


    return correct/num




def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset_dir', type=str, default='F:\\download\\CVS_Dataset_New\\', help='the path of the dataset')
    parser.add_argument('--batch_size', type=int, default=3,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='training batch size')
    parser.add_argument('--epoch',type=int,default=2000,help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='0', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    parser.add_argument('--image_net_weights', type=str, default='../model/pretrain_image_model.pkl', help='image net weights')
    parser.add_argument('--audio_net_weights', type=str, default='../model/pretrain_audio_model.pt',
                        help='image net weights')

    args = parser.parse_args()

    root_dir = args.dataset_dir

    train_dataset = CVSDataset(root_dir,'train')
    test_dataset = CVSDataset(root_dir,'test') # 未来优化的时候加入验证集

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



    # 定义网络
    image_net = IMG_NET(num_classes= 30) #原始网络是30
    gpu_ids = [int(e) for e in args.gpu_ids.split(',')]
    image_net_cuda = t.nn.DataParallel(image_net, device_ids=gpu_ids).cuda()

    state = pickle.load(open(args.image_net_weights,mode='rb'))
    image_net_cuda.load_state_dict(state)

    audio_net = AUD_NET()

    state = t.load(args.audio_net_weights)['model']
    audio_net.load_state_dict(state)
    gpu_ids = [int(e) for e in args.gpu_ids.split(',')] # 注意顺序
    audio_net_cuda = t.nn.DataParallel(audio_net, device_ids=gpu_ids).cuda()

    # all stand up
    fusion_net = FUS_NET(image_net_cuda,audio_net_cuda,num_classes=17)
    gpu_ids = [int(e) for e in args.gpu_ids.split(',')]
    fusion_net_cuda = t.nn.DataParallel(fusion_net, device_ids=gpu_ids).cuda()


    loss_func = t.nn.CrossEntropyLoss()

    optimizer = optim.SGD(params=fusion_net_cuda.parameters(), lr=args.learning_rate, momentum=args.momentum)

    max_acc = 0.
    for e in range(args.epoch):
        net_train(fusion_net_cuda,train_dataloader,optimizer,loss_func,e,args)
        acc = net_test(fusion_net_cuda,test_dataloader)
        print("EPOCH:%d ACC:%.4f" % (e,acc))

        if(acc > max_acc):
            max_acc = acc
            # saing... TODO:: save model
            net_state =fusion_net_cuda.state_dict()
            pickle.dump(net_state,open('../model/fusion_net_model.pkl',mode='wb'))


if __name__ == '__main__':
    main()






