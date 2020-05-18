import torch
import torch.nn as nn


class FusionNet(nn.Module):

    def __init__(self, image_net, audio_net,num_classes):
        super(FusionNet, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        self.num_classes = num_classes

        self.image_fc1 = nn.Linear(2048,1024)
        self.audio_fc1 = nn.Linear(2048,1024)

        self.fusion_fc1 = nn.Linear(2048,512)
        self.fusion_fc2 = nn.Linear(512,self.num_classes)

    def forward(self, image,audio):

        image_rep = self.image_net(image)[0] # fc_rep
        audio_rep = self.audio_net(audio)[0] # fc_rep

        image_rep = self.image_fc1(image_rep)
        image_rep = torch.sigmoid(image_rep) # batch_size * 1024

        audio_rep = self.audio_fc1(audio_rep)
        audio_rep = torch.sigmoid(audio_rep) # batch_Size * 1024

        # concat

        concat_rep = torch.cat((image_rep,audio_rep),dim = 1)

        concat_rep = self.fusion_fc1(concat_rep)
        concat_rep = torch.sigmoid(concat_rep)
        concat_rep = self.fusion_fc2(concat_rep) # outter cross-entropy + softmax

        concat_rep = torch.sigmoid(concat_rep)

        return concat_rep



