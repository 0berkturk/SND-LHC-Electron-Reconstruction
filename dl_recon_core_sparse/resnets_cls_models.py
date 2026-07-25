import torch
import torch.nn as nn
import torch.nn.functional as F
import config



class Classifier_mlp(nn.Module):
    def __init__(self, dim, hidden_dim,out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
            nn.GELU(),
            nn.Linear(8,out_dim )
        )

    def forward(self, image):
        if config.IS_BINARY:
            x=torch.flatten(self.network(image))
        else:
            x = self.network(image)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(1,3) ,stride=(1,2), padding=(0,1)):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_sizes, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(3,3), stride=1, padding=(1,1), bias=False) ## size shouldnt change at this layer
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample when input and output sizes differ
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            if stride[1]!=stride[0]:
                stride=(stride[0], stride[1])

            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        #rint("1in",out.shape)
        out = self.bn2(self.conv2(out))
        #print("2in",out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)
            #print("3in",identity.shape)

        out += identity
        out = F.relu(out)
        return out

class ResNets_scifi(nn.Module):
    def __init__(self, in_chan, num_classes=4):
        super(ResNets_scifi, self).__init__()
        # kernel 3, stride 2, padding 1 -> shape is halved.
        # kernel 1, stride 1, padding 0 -> shape is constant, better
        # kernel 3, stride 1, padding 1 -> shape is constant.
        mult=4
        # Input: (B, 1, 5, 1536)
        self.layer1 = ResidualBlock(in_chan, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 32, 5, 768)
        self.layer2 = ResidualBlock(32*mult, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 64, 5, 384)
        self.layer3 = ResidualBlock(32*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 192)
        self.layer4 = ResidualBlock(64*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer5 = ResidualBlock(64 * mult, 128 * mult, kernel_sizes=(3, 4), stride=(2, 2), padding=(1, 1))  # -> (B, 128, 5, 96)
        self.layer6 = ResidualBlock(128 * mult, 128 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer7 = ResidualBlock(128 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer8 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer9 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)




        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global pooling
        self.fc = nn.Sequential(nn.Linear(1024,128),nn.ReLU(),
                                #nn.Linear(512,128),nn.ReLU(),
                                nn.Linear(128,num_classes))#nn.Linear(1280, num_classes)
        print(num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.pool(out).reshape(-1,1024)
        out = self.fc(out)
        return out


class ResNets_scifi_R256Optimized_2layer(nn.Module):
    def __init__(self, in_chan, num_classes=4):
        super(ResNets_scifi_R256Optimized_2layer, self).__init__()
        
        # mult=4 makes the channel progression: 128 -> 128 -> 256 -> 256 -> 512 -> 1024
        mult = 4 
        
        # Layer 1: (B, C, 5, 512) -> (B, 128, 5, 256)
        self.layer1 = ResidualBlock(in_chan, 32*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 2: (B, 128, 5, 256) -> (B, 128, 5, 128)
        self.layer2 = ResidualBlock(32*mult, 32*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 3: (B, 128, 5, 128) -> (B, 256, 5, 64)
        self.layer3 = ResidualBlock(32*mult, 64*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 4: (B, 256, 5, 64) -> (B, 256, 3, 32) -- Reducung Height here
        self.layer4 = ResidualBlock(64*mult, 64*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 5: (B, 256, 3, 32) -> (B, 512, 2, 16)
        self.layer5 = ResidualBlock(64*mult, 128*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 6: (B, 512, 2, 16) -> (B, 1024, 1, 8)
        self.layer6 = ResidualBlock(128*mult, 256*mult, kernel_sizes=(3,3), stride=(2,2), padding=(1,1))  

        # Global Pooling collapses (1, 8) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.fc = nn.Sequential(
            nn.Linear(256 * mult, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        out = self.layer1(x)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        out = self.layer4(out)
        
        out = self.layer5(out)
        
        out = self.layer6(out)
        
        
        out = self.pool(out)
        out = torch.flatten(out, 1) 
        out = self.fc(out)

        return out

class ResNets_scifi_R256Optimized(nn.Module):
    def __init__(self, in_chan, num_classes=4):
        super(ResNets_scifi_R256Optimized, self).__init__()
        
        # mult=4 makes the channel progression: 128 -> 128 -> 256 -> 256 -> 512 -> 1024
        mult = 4 
        
        # Layer 1: (B, C, 5, 512) -> (B, 128, 5, 256)
        self.layer1 = ResidualBlock(in_chan, 32*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 2: (B, 128, 5, 256) -> (B, 128, 5, 128)
        self.layer2 = ResidualBlock(32*mult, 32*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 3: (B, 128, 5, 128) -> (B, 256, 5, 64)
        self.layer3 = ResidualBlock(32*mult, 64*mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  
        
        # Layer 4: (B, 256, 5, 64) -> (B, 256, 3, 32) -- Reducung Height here
        self.layer4 = ResidualBlock(64*mult, 64*mult, kernel_sizes=(3,3), stride=(2,2), padding=(1,1))  
        
        # Layer 5: (B, 256, 3, 32) -> (B, 512, 2, 16)
        self.layer5 = ResidualBlock(64*mult, 128*mult, kernel_sizes=(3,3), stride=(2,2), padding=(1,1))  
        
        # Layer 6: (B, 512, 2, 16) -> (B, 1024, 1, 8)
        self.layer6 = ResidualBlock(128*mult, 256*mult, kernel_sizes=(3,3), stride=(2,2), padding=(1,1))  

        # Global Pooling collapses (1, 8) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.fc = nn.Sequential(
            nn.Linear(256 * mult, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        
        out = self.pool(out)
        out = torch.flatten(out, 1) 
        out = self.fc(out)
        return out

class ResNets_scifi_256R(nn.Module):
    def __init__(self, in_chan, num_classes=4):
        super(ResNets_scifi_256R, self).__init__()
        # kernel 3, stride 2, padding 1 -> shape is halved.
        # kernel 1, stride 1, padding 0 -> shape is constant, better
        # kernel 3, stride 1, padding 1 -> shape is constant.
        mult=4
        # Input: (B, 1, 5, 512)
        self.layer1 = ResidualBlock(in_chan, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 32, 5, 256)
        self.layer2 = ResidualBlock(32*mult, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 64, 5, 128)
        self.layer3 = ResidualBlock(32*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 64)
        self.layer4 = ResidualBlock(64*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 32)
        self.layer5 = ResidualBlock(64 * mult, 128 * mult, kernel_sizes=(3, 4), stride=(2, 2), padding=(1, 1))  # -> (B, 128, 2, 16)
        self.layer6 = ResidualBlock(128 * mult, 128 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 8)
        self.layer7 = ResidualBlock(128 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 4)
        self.layer8 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 2)
        self.layer9 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 1)




        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global pooling
        self.fc = nn.Sequential(nn.Linear(1024,128),nn.ReLU(),
                                #nn.Linear(512,128),nn.ReLU(),
                                nn.Linear(128,num_classes))#nn.Linear(1280, num_classes)
        print(num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.pool(out).reshape(-1,1024)
        out = self.fc(out)
        return out



class ResNets(nn.Module):
    def __init__(self, in_chan, num_classes=4):
        super(ResNets, self).__init__()
        # kernel 3, stride 2, padding 1 -> shape is halved.
        # kernel 1, stride 1, padding 0 -> shape is constant, better
        # kernel 3, stride 1, padding 1 -> shape is constant.
        mult=4
        # Input: (B, 1, 5, 1536)
        self.layer1 = ResidualBlock(in_chan, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 32, 5, 768)
        self.layer2 = ResidualBlock(32*mult, 32*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 64, 5, 384)
        self.layer3 = ResidualBlock(32*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 192)
        self.layer4 = ResidualBlock(64*mult, 64*mult, kernel_sizes=(3,4), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer5 = ResidualBlock(64 * mult, 128 * mult, kernel_sizes=(3, 4), stride=(2, 2), padding=(1, 1))  # -> (B, 128, 5, 96)
        self.layer6 = ResidualBlock(128 * mult, 128 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer7 = ResidualBlock(128 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer8 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)
        self.layer9 = ResidualBlock(256 * mult, 256 * mult, kernel_sizes=(3,3), stride=(1,2), padding=(1,1))  # -> (B, 128, 5, 96)


        #ds is 2x4x60
        self.layer_ds1=ResidualBlock(2, 32,kernel_sizes=(3,3), stride=(1,2), padding=(1,1))# 4x30
        self.layer_ds2 = ResidualBlock(32, 64, kernel_sizes=(3, 3), stride=(2, 2), padding=(1, 1))  # 2x15
        self.layer_ds3 = ResidualBlock(64, 128, kernel_sizes=(3, 3), stride=(2, 2), padding=(1, 1))  # 1x8
        self.layer_ds4 = ResidualBlock(128, 128, kernel_sizes=(3, 3), stride=(1, 1), padding=(1, 1))  # 5x10
        #self.layer_ds5 = ResidualBlock(128, 128, kernel_sizes=(3, 3), stride=(1, 1), padding=(1, 1))  # 5x10


        #ds is 2x5x10
        self.layer_us1=ResidualBlock(2, 32,kernel_sizes=(3,3), stride=(1,1), padding=(1,1))# 5x10
        self.layer_us2 = ResidualBlock(32, 64, kernel_sizes=(3, 3), stride=(1, 1), padding=(1, 1))  # 5x10
        self.layer_us3 = ResidualBlock(64, 128, kernel_sizes=(3, 3), stride=(2, 2), padding=(1, 1))  # 3x6
        #self.layer_us4 = ResidualBlock(128, 128, kernel_sizes=(3, 3), stride=(1, 1), padding=(1, 1))  # 5x10
        #self.layer_us5 = ResidualBlock(128, 128, kernel_sizes=(3, 3), stride=(1, 1), padding=(1, 1))  # 5x10



        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global pooling
        self.fc = nn.Sequential(nn.Linear(1280,128),nn.ReLU(),
                                #nn.Linear(512,128),nn.ReLU(),
                                nn.Linear(128,num_classes))#nn.Linear(1280, num_classes)

    def forward(self, input_x):
        scifi,us,ds=input_x
        x=scifi
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.pool(out)
        out = torch.flatten(out, 1) ## Nx256
        out_ds=self.layer_ds1(ds)
        #print("1",out_ds.shape)
        out_ds=self.layer_ds2(out_ds)
        #print("2",out_ds.shape)
        out_ds=self.layer_ds3(out_ds)
        out_ds=self.layer_ds4(out_ds)
        #out_ds=self.layer_ds5(out_ds)
        #print("3",out_ds.shape)
        out_ds=self.pool(out_ds)
        #print(out_ds.shape)
        out_ds=torch.flatten(out_ds, 1)
        #print("4",out_ds.shape)


        out_us = self.layer_us1(us)
        #print("5",out_us.shape)
        out_us=self.layer_us2(out_us)
        #print("6",out_us.shape)
        out_us=self.layer_us3(out_us)
        #out_us=self.layer_us4(out_us)
        #out_us=self.layer_us5(out_us)
        #print("7",out_us.shape)
        out_us=self.pool(out_us)
        #print("8",out_us.shape)
        out_us=torch.flatten(out_us, 1)
        #print("9",out_us.shape)

        #print(out.shape, out_ds.shape, out_us.shape)
        out=torch.cat([out, out_ds, out_us],1)
        #
        out = self.fc(out)
        return out
