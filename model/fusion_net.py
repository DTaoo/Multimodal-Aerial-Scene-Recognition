import torch
import torch.nn as nn
import numpy as np

class FusionNet(nn.Module):

    def __init__(self, image_net, audio_net,num_classes):
        super(FusionNet, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        #self.image_fc1 = nn.Linear(2048,1024)
        #self.audio_fc1 = nn.Linear(2048,1024)

        #self.fusion_fc1 = nn.Linear(2048,512)
        #self.fusion_fc2 = nn.Linear(512,self.num_classes)
        self.fusion_fc  = nn.Linear(4096, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, audio):

        image_rep = self.image_net(image)[0] # fc_rep
        audio_rep = self.audio_net(audio)[0] # fc_rep
        #audio_rep = torch.zeros_like(image_rep)

        #image_rep = self.image_fc1(image_rep)
        #image_rep = self.relu(image_rep) # batch_size * 1024

        #audio_rep = self.audio_fc1(audio_rep)
        #audio_rep = self.relu(audio_rep) # batch_Size * 1024

        concat_rep = torch.cat((image_rep,audio_rep), dim = 1)

        concat_rep  = self.fusion_fc(concat_rep)
        return concat_rep

class FusionNet_Bayes(nn.Module):

    def __init__(self, image_net, audio_net, num_classes):
        super(FusionNet_Bayes, self).__init__()

        #self.scene_to_event = np.load('scene_to_event_prior_59.npy')
        #self.scene_to_event = torch.from_numpy(self.scene_to_event).cuda()
        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        #self.image_fc1 = nn.Linear(2048,1024)
        #self.audio_fc1 = nn.Linear(2048,1024)

        #self.fusion_fc1 = nn.Linear(2048,512)
        #self.fusion_fc2 = nn.Linear(512,self.num_classes)
        self.fusion_fc  = nn.Linear(4096, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, audio):

        image_rep = self.image_net(image)[0] # fc_rep
        audio_rep = self.audio_net(audio)[0] # fc_rep

        concat_rep = torch.cat((image_rep,audio_rep),dim = 1)

        concat_rep  = self.fusion_fc(concat_rep)
        '''
        scene_predict = self.softmax(concat_rep)
        scene_to_event_ = torch.squeeze(scene_to_event[0,:,:])
        #print(scene_to_event_.shape)
        event_predict = scene_predict.mm(scene_to_event_)
        '''
        return concat_rep


class FusionNet_SQ(nn.Module):

    def __init__(self, image_net, audio_net,num_classes):
        super(FusionNet_SQ, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        #self.image_fc1 = nn.Linear(2048,1024)
        #self.audio_fc1 = nn.Linear(2048,1024)

        #self.fusion_fc1 = nn.Linear(2048,512)
        #self.fusion_fc2 = nn.Linear(512,self.num_classes)
        self.fusion_fc  = nn.Linear(4096, self.num_classes)
        self.KD_fc = nn.Linear(2048, 527)
        self.sig = nn.Sigmoid()
      

    def forward(self, image,audio):

        image_rep = self.image_net(image)[0] # fc_rep
        audio_rep = self.audio_net(audio)[0] # fc_rep
        #audio_rep = torch.zeros_like(image_rep)

        concat_rep = torch.cat((image_rep,audio_rep),dim = 1)

        concat_rep  = self.fusion_fc(concat_rep)
        sed_output  = self.KD_fc(audio_rep)
        #sed_output  = self.sig(sed_output)
        return concat_rep, sed_output



class FusionNet_KL(nn.Module):

    def __init__(self, image_net, audio_net,num_classes):
        super(FusionNet_KL, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        #self.image_fc1 = nn.Linear(2048,1024)
        #self.audio_fc1 = nn.Linear(2048,1024)

        #self.fusion_fc1 = nn.Linear(2048,512)
        #self.fusion_fc2 = nn.Linear(512,self.num_classes)
        self.fusion_fc  = nn.Linear(4096, self.num_classes)
        self.KD_fc = nn.Linear(4096, 527)
        self.sig = nn.Sigmoid()
         

    def forward(self, image,audio):

        image_rep = self.image_net(image)[0] # fc_rep
        audio_rep = self.audio_net(audio)[0] # fc_rep
        #image_rep = torch.zeros_like(audio_rep)

        concat_rep = torch.cat((image_rep,audio_rep),dim = 1)
        concat_rep_ = self.fusion_fc(concat_rep)
        sed_output  = self.KD_fc(concat_rep)
        sed_output  = self.sig(sed_output)
        return concat_rep_, sed_output


class FusionNet_uni(nn.Module):

    def __init__(self, image_net, audio_net,num_classes):
        super(FusionNet_uni, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        #self.image_fc1 = nn.Linear(2048,1024)
        #self.audio_fc1 = nn.Linear(2048,1024)

        #self.fusion_fc1 = nn.Linear(2048,512)
        #self.fusion_fc2 = nn.Linear(512,self.num_classes)
        self.fusion_fc  = nn.Linear(2048, self.num_classes)
        self.KD_fc = nn.Linear(2048, 527)
        self.sig = nn.Sigmoid()

    def forward(self, image):

        image_rep = self.image_net(image)[0] # fc_rep
        #audio_rep = self.audio_net(audio)[0] # fc_rep

        #concat_rep = torch.cat((image_rep),dim = 1)

        concat_rep_  = self.fusion_fc(image_rep)
        sed_output  = self.KD_fc(image_rep)
        sed_output = self.sig(sed_output)
        return concat_rep_, sed_output

