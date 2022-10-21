import torch
import torch.nn as nn
from models.modules.SCRAM import Model as SCRAM

class Model(nn.Module):

    def __init__(self, nChannel=6,nFeat=32):
        super(Model, self).__init__()


        # Encoder
        self.conv1 = nn.Conv2d(nChannel, nFeat// 2, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(nFeat // 2, nFeat, kernel_size=3, padding=1, stride=2)

        # Spatial attention
        self.att11 = SCRAM()

        # Spatial attention
        self.att31 = SCRAM()

        self.conv3 = operation(nFeat, nFeat * 2, kernel_size=3, padding=1, stride=1)

        # MERGE
        self.conv4 = operation(nFeat * 4, nFeat * 2, kernel_size=3, padding=1)
        self.conv5 = operation(nFeat *  2, nFeat *  2, kernel_size=3, padding=1)
        self.decoder = operation(nFeat *  2, nFeat *  2, kernel_size=3, padding=1, stride=1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.decoder3 = operation(nFeat // 2, 3, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x1, x2, x3):

        # STEM
        F1_ = self.relu(self.conv1(x1))
        F2_ = shortcut = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))


        F1_ = self.relu(self.conv2(F1_))
        F2_ = self.relu(self.conv2(F2_))
        F3_ = self.relu(self.conv2(F3_))

        # Attention low/middle
        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.sigmoid(self.att11(F1_i))
        F1_ *= F1_A

        # Attention low/middle
        F3_i = torch.cat((F3_, F2_), 1)
        F3_A =self.sigmoid(self.att31(F3_i))
        F3_ *= F3_A


        F1_ = self.relu(self.conv3(F1_))
        F2_b = self.relu(self.conv3(F2_))
        F3_ = self.relu(self.conv3(F3_))


        # Merger
        F13_ = torch.cat((F1_, F3_), 1)
        F13_ = self.relu(self.conv4(F13_))
        F13_ += F2_b

        F_ = self.relu(self.conv5(F13_))
        F_ = self.relu(self.decoder(F_))

        # Decoder
        F_ = self.pixel_shuffle(F_)
        F_ += shortcut
        F_ = self.sigmoid(self.decoder3(F_))


        return F_
