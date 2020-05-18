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
from fusion_net import FusionNet, FusionNet_unimodal
from fusion_net import FusionNet_Bayes
from fusion_net import FusionNet_KD
import pickle
import torch.optim as optim
import datetime
import os
import sklearn
from data_partition import data_construction
import numpy as np
from torch import nn


def net_test(net,data_loader, scene_to_event, iteration, cal_map=False):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    predict_labels = np.array([])
    ground_labels  = np.array([])
    predict_events = np.array([]).reshape(0,527)
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            img, aud, label, _e, _r = data
            sample_num = img.shape[0]
            img, aud, label = img.type(torch.FloatTensor).cuda(), aud.type(torch.FloatTensor).cuda(), label.type(torch.LongTensor).cuda() # gpu
            
            #scene_to_event_ = np.tile(scene_to_event, (sample_num,1,1))
            #scene_to_event_cuda = torch.from_numpy(scene_to_event_).cuda()
            
            output = net(img, aud)
            scene_prob   = torch.nn.functional.softmax(output, dim=1)
            event_output  = scene_prob.mm(scene_to_event)
            event_output_ = event_output.cpu().numpy()

            predict_label = torch.argmax(output, dim=1)
            #if predict_labels == None:
            #    predict_labels = predict_label.cpu().numpy()
            #else:
            predict_labels = np.concatenate((predict_labels, predict_label.cpu().numpy()))
            predict_events = np.concatenate((predict_events, event_output_))
            #if ground_labels == None:
            #    ground_labels = label.cpu().numpy()
            #else:
            ground_labels = np.concatenate((ground_labels, label.cpu().numpy()))
            correct += ((predict_label == label).sum().cpu().numpy())
            # map

    results = sklearn.metrics.classification_report(ground_labels, predict_labels, digits=4)
    np.save('visual/bayes_label_%d.npy'%iteration, ground_labels)
    np.save('visual/bayes_predcit_event_%d.npy'%iteration, predict_events)
    
    (precision, recall, fscore, sup) = sklearn.metrics.precision_recall_fscore_support(ground_labels, predict_labels, average='weighted')
    confusion_matrix = sklearn.metrics.confusion_matrix(ground_labels, predict_labels)
    np.save('visual/bayes_confusion_%d.npy'%iteration, confusion_matrix)
    acc = correct/num

    return (acc, precision, recall, fscore, results)


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

class cosine_loss(torch.nn.Module):
    def __init__(self):
        super(cosine_loss, self).__init__()
        self.cosine = torch.nn.CosineSimilarity(dim=1)

    def forward(self, pred, true):
        return torch.mean(1. - self.cosine(pred, true))
        

def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--dataset_dir', type=str, default='F:\\download\\CVS_Dataset_New\\', help='the path of the dataset')
    parser.add_argument('--batch_size', type=int, default=64,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='training batch size')
    parser.add_argument('--epoch',type=int,default=2000,help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    parser.add_argument('--image_net_weights', type=str, default='AID_visual_pretrain.pt', help='image net weights')
    parser.add_argument('--audio_net_weights', type=str, default='audioset_audio_pretrain.pt',
                        help='audio net weights')

    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/soundscape/data/',
                        help='image net weights')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--data_name', type=str, default='CVS_data_ind.pkl')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--audionet_pretrain', type=int, default=1)
    parser.add_argument('--videonet_pretrain', type=int, default=1) 
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--reg_weight', type=float, default=0.001)

    parser.add_argument('--using_event_knowledge', default=True, action='store_true')
    parser.add_argument('--using_event_regularizer', default=True, action='store_true')
    
    args = parser.parse_args()

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_construction(args.data_dir)

    #f = open(args.data_name, 'wb')
    #data = {'train_sample':train_sample, 'train_label':train_label, 'test_sample':test_sample, 'test_label':test_label}
    #pickle.dump(data, f)
    #f.close()

    print('bayes model...')
    print(args.videonet_pretrain)
    print(args.audionet_pretrain)
    print(args.seed)
    print(args.kd_weight)
    print(args.reg_weight)
    print(args.using_event_knowledge)
    print(args.using_event_regularizer)
    print(args.learning_rate)

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
    fusion_net = FusionNet(image_net, audio_net, num_classes=13)

    gpu_ids = [i for i in range(4)]
    fusion_net_cuda = torch.nn.DataParallel(fusion_net, device_ids=gpu_ids).cuda()

    loss_func_CE  = torch.nn.CrossEntropyLoss()
    loss_func_BCE = torch.nn.BCELoss(reduce=True)
    loss_func_COSINE = cosine_loss()
    softmax_ = nn.LogSoftmax(dim=1)
    loss_func_KL   = torch.nn.KLDivLoss()

    optimizer = optim.Adam(params=fusion_net_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0001)

    max_fscore = 0.

    scene_to_event = np.load('scene_to_event_prior_59.npy')
    #scene_to_event = np.expand_dims(scene_to_event, 0)
    
    scene_to_event = torch.from_numpy(scene_to_event).cuda()
    #scene_to_event = torch.unsqueeze(x, 0)
    #scene_to_event = scene_to_event.repeat(64,1)
    count = 0
    for e in range(args.epoch):

        fusion_net_cuda.train()
        begin_time = datetime.datetime.now()

        scene_loss = 0.
        event_loss = 0.
        regu_loss  = 0.
        batch_num = int(len(train_dataloader.dataset) / args.batch_size)

        for i, data in enumerate(train_dataloader, 0):
            # print('batch:%d/%d' % (i,batch_num))
            img, aud, scene_label, event_label, event_corr = data
            sample_num = img.shape[0]
            img, aud, scene_label, event_label, event_corr = img.type(torch.FloatTensor).cuda(), aud.type(torch.FloatTensor).cuda(), scene_label.type(
                torch.LongTensor).cuda(), event_label.type(torch.FloatTensor).cuda(), event_corr.type(torch.FloatTensor).cuda()

            #scene_to_event = np.expand_dims(scene_to_event, 0)
            #scene_to_event_ = np.tile(scene_to_event, (sample_num,1,1))
            #scene_to_event_cuda = torch.from_numpy(scene_to_event_).cuda()

            optimizer.zero_grad()
            
            scene_output = fusion_net_cuda(img, aud)

            CE_loss  = loss_func_CE(scene_output, scene_label)

            scene_loss += CE_loss.cpu()

            if args.using_event_knowledge:
                scene_prob   = torch.nn.functional.softmax(scene_output, dim=1)
                event_output  = scene_prob.mm(scene_to_event)           
                
                kl_loss = loss_func_BCE(event_output, event_label) * args.kd_weight
                #cosine_loss_ = loss_func_COSINE(event_output, event_label) * args.kd_weight
                event_loss += kl_loss.cpu()

                if args.using_event_regularizer:
                    #print('tt')
                    #regularizer_loss = loss_func_KL(softmax_(event_output), softmax_(event_label))
                    regularizer_loss = loss_func_COSINE(event_output, event_corr) * args.kd_weight * args.reg_weight
                    losses = CE_loss + kl_loss + regularizer_loss 
                    regu_loss += regularizer_loss.cpu()                   
                else:
                   
                    losses = CE_loss + kl_loss
            else:
                losses = CE_loss

            losses.backward()
            optimizer.step()

        end_time = datetime.datetime.now()
        delta_time = (end_time - begin_time)
        delta_seconds = (delta_time.seconds * 1000 + delta_time.microseconds) / 1000

        (val_acc, val_precision, val_recall, val_fscore, _) = net_test(fusion_net_cuda, val_dataloader, scene_to_event, e)
        print('epoch:%d scene loss:%.4f event loss:%.4f reg loss: %.4f val acc:%.4f val_precision:%.4f val_recall:%.4f val_fscore:%.4f ' % (e, scene_loss.cpu(), event_loss.cpu(), regu_loss.cpu(), val_acc, val_precision, val_recall, val_fscore))
        if val_fscore > max_fscore:
            count = 0
            max_fscore = val_fscore
            (test_acc, test_precision, test_recall, test_fscore, results) = net_test(fusion_net_cuda, test_dataloader, scene_to_event,e)
            #print(results)
            test_acc_list = [test_acc]
            test_precision_list = [test_precision]
            test_recall_list = [test_recall]
            test_fscore_list = [test_fscore]
            print('mark...') 
            #print('test acc:%.4f precision:%.4f recall:%.4f fscore:%.4f' % (test_acc, test_precision, test_recall, test_fscore))
            
        else:
            count = count + 1
            (test_acc, test_precision, test_recall, test_fscore, results) = net_test(fusion_net_cuda, test_dataloader, scene_to_event,e)
            #print(results)            
            
            test_acc_list.append(test_acc)
            test_precision_list.append(test_precision)
            test_recall_list.append(test_recall)
            test_fscore_list.append(test_fscore)
        
        if count == 5:
            test_acc_mean = np.mean(test_acc_list)
            test_acc_std  = np.std(test_acc_list)

            test_precision_mean = np.mean(test_precision_list)
            test_precision_std  = np.std(test_precision_list)

            test_recall_mean = np.mean(test_recall_list)
            test_recall_std = np.std(test_recall_list)

            test_fscore_mean = np.mean(test_fscore_list)
            test_fscore_std = np.std(test_fscore_list)
 
            print('test acc:%.4f (%.4f) precision:%.4f (%.4f) recall:%.4f (%.4f) fscore:%.4f(%.4f)' % (test_acc_mean, test_acc_std, test_precision_mean, test_precision_std, test_recall_mean, test_recall_std, test_fscore_mean, test_fscore_std))
            count = 0
            test_acc_list = []
            test_precision_list = []
            test_recall_list = []
            test_fscore_list = []
            # Save model
            MODEL_PATH = 'checkpoint2'
            MODEL_FILE = os.path.join(MODEL_PATH, 'bayes_checkpoint%d_%.3f.pt' % (e,test_fscore_mean))
            state = {'model': fusion_net_cuda.state_dict(), 'optimizer': optimizer.state_dict()}
            sys.stderr.write('Saving model to %s ...\n' % MODEL_FILE)
            torch.save(state, MODEL_FILE) 

        if e in [30, 60, 90]:
            decrease_learning_rate(optimizer, 0.1)
            print('decreased learning rate by 0.1')


if __name__ == '__main__':
    main()






