import torch
import torch.nn as nn
import torch.nn.functional as F

class ISCX_module(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,1))
        self.flatten1 = nn.Flatten()

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,2))
        self.flatten2 = nn.Flatten()

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,4))
        self.flatten3 = nn.Flatten()

        self.conv4 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2,2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))
        self.flatten4 = nn.Flatten()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2))
        self.flatten5 = nn.Flatten()
        
        self.fc = nn.Linear(131776, num_classes)

        
    def forward(self,x):
        out1 = self.conv1(x)
        out1 = F.relu(out1)
        out1 = self.flatten1(out1)


        out2 = self.conv2(x)
        out2 = F.relu(out2)
        out2 = self.flatten2(out2)


        out3 = self.conv3(x)
        out3 = F.relu(out3)
        out3 = self.flatten3(out3)


        out4 = self.conv4(x)
        out4 = F.relu(out4)
        out4 = self.maxpool4(out4)

        out5 = self.conv5(out4)
        out5 = F.relu(out5)

        out4 = self.flatten4(out4)
        out5 = self.flatten5(out5)


        # print("out1.size", out1.size())
        # print("out2.size", out2.size())
        # print("out3.size", out3.size())
        # print("out4.size", out4.size())
        # print("out5.size", out5.size())


        out = torch.cat((out1,out2,out3,out4,out5),1)
        return out


class ISCX_LeNet(nn.Module):
    def __init__(self, channel=3, hideen=1128, num_classes=10):
        super(ISCX_LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        # ISCX_LeNet x torch.Size([1, 3, 375, 4])
        # print("ISCX_LeNet x", x.shape)
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
