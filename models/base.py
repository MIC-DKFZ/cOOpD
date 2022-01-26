import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import SimpleITK as sitk

class CNN3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_feat, bias=True, num_layers=4):
        """Baseline CNN architecture
            Outputs of shape out_channelsx1x1x1

        Args:
            in_channels ([type]): [channels of input]
            out_channles ([type]): [output size]
            num_feat ([type]): [feature channels of 1st conv]
        """
        super(CNN3D, self).__init__()

        self.l1 = nn.Sequential(nn.Conv3d(in_channels, num_feat, kernel_size=3, stride=2, bias=bias),
                                nn.Conv3d(num_feat, num_feat, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.BatchNorm3d(num_feat),
                                nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv3d(num_feat, num_feat*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.Conv3d(num_feat*2, num_feat*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.BatchNorm3d(num_feat*2),
                                nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv3d(num_feat*2, num_feat*4, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.Conv3d(num_feat*4, num_feat*4, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.BatchNorm3d(num_feat*4),
                                nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv3d(num_feat*4, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                                nn.BatchNorm3d(out_channels),
                                nn.ReLU())

        self.z_dim = out_channels




    def forward(self, x):

        #sitk.WriteImage(sitk.GetImageFromArray(x.detach().cpu().numpy().astype('float32')[0,0,:,:,:]), '/home/silvia/Downloads/ex_original.nii.gz')
        out = self.l1(x)

        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,2,:,:,:]), '/home/silvia/Downloads/ex_layer1_03.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,4,:,:,:]), '/home/silvia/Downloads/ex_layer1_05.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,6,:,:,:]), '/home/silvia/Downloads/ex_layer1_07.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,8,:,:,:]), '/home/silvia/Downloads/ex_layer1_09.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,10,:,:,:]), '/home/silvia/Downloads/ex_layer1_11.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,12,:,:,:]), '/home/silvia/Downloads/ex_layer1_13.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,14,:,:,:]), '/home/silvia/Downloads/ex_layer1_15.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,15,:,:,:]), '/home/silvia/Downloads/ex_layer1_16.nii.gz')

        out = self.l2(out)

        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,2,:,:,:]), '/home/silvia/Downloads/ex_layer2_03.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,4,:,:,:]), '/home/silvia/Downloads/ex_layer2_05.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,6,:,:,:]), '/home/silvia/Downloads/ex_layer2_07.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,8,:,:,:]), '/home/silvia/Downloads/ex_layer2_09.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,10,:,:,:]), '/home/silvia/Downloads/ex_layer2_11.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,12,:,:,:]), '/home/silvia/Downloads/ex_layer2_13.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,14,:,:,:]), '/home/silvia/Downloads/ex_layer2_15.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,15,:,:,:]), '/home/silvia/Downloads/ex_layer2_16.nii.gz')


        out = self.l3(out)

        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,2,:,:,:]), '/home/silvia/Downloads/ex_layer3_03.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,4,:,:,:]), '/home/silvia/Downloads/ex_layer3_05.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,6,:,:,:]), '/home/silvia/Downloads/ex_layer3_07.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,8,:,:,:]), '/home/silvia/Downloads/ex_layer3_09.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,10,:,:,:]), '/home/silvia/Downloads/ex_layer3_11.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,12,:,:,:]), '/home/silvia/Downloads/ex_layer3_13.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,14,:,:,:]), '/home/silvia/Downloads/ex_layer3_15.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,15,:,:,:]), '/home/silvia/Downloads/ex_layer3_16.nii.gz')


        out = self.l4(out)
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,2,:,:,:]), '/home/silvia/Downloads/ex_layer4_03.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,4,:,:,:]), '/home/silvia/Downloads/ex_layer4_05.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,6,:,:,:]), '/home/silvia/Downloads/ex_layer4_07.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,8,:,:,:]), '/home/silvia/Downloads/ex_layer4_09.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,10,:,:,:]), '/home/silvia/Downloads/ex_layer4_11.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,12,:,:,:]), '/home/silvia/Downloads/ex_layer4_13.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,14,:,:,:]), '/home/silvia/Downloads/ex_layer4_15.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(out.detach().cpu().numpy().astype('float32')[0,15,:,:,:]), '/home/silvia/Downloads/ex_layer4_16.nii.gz')
        return out

    def features(self, x):
        out = []
        out.append(self.l1(x))
        out.append(self.l2(out[-1]))
        out.append(self.l3(out[-1]))
        out.append(self.l4(out[-1]))
        return out


class BCNN3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_feat, bias=True, num_layers=4):
        """Baseline CNN architecture
            Outputs of shape out_channelsx1x1

        Args:
            in_channels ([type]): [channels of input]
            out_channles ([type]): [output size]
            num_feat ([type]): [feature channels of 1st conv]
        """
        super(BCNN3D, self).__init__()

        layers = [nn.Conv3d(in_channels, num_feat, kernel_size=4, stride=2, padding=1, bias=bias)]
        print(range(num_layers - 2))
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.BatchNorm3d(num_feat), nn.ReLU(),
                                        nn.Conv3d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1,
                                                  bias=bias)))
            num_feat = num_feat * 2
        layers.append(
            nn.Sequential(nn.BatchNorm3d(num_feat), nn.ReLU(), nn.Conv3d(num_feat, out_channels, 3, 1, 0, bias=bias))) #4

        self.layers = nn.ModuleList(layers)

        self.z_dim = out_channels



    def forward(self, x):
        # # test = self.test(x)
        # # test = self.test_1(test)
        # # test = self.max(test)
        # # test = self.norm(test)
        # # test = self.test_2(test)
        # out = self.l1(x)
        # out = self.l2(out)
        # #out = self.l3(out)
        # print(out.size(0), out.size(1), out.size(2))
        # # out = out.view(-1, out.size(0)*out.size(1)*out.size(2)*out.size(3)*out.size(4))
        # # out = F.relu(self.linear(out))
        #
        # #out = self.linear(out)
        # #out = self.l3(out)
        # return out

        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x)
            else:
                out = layer(out)
        return out

    def features(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = [layer(x)]
            else:
                out.append(layer(out)[-1])
        return out

    def features(self, x):
        out = []
        out.append(self.l1(x))
        out.append(self.l2(out[-1]))
        #out.append(self.l3(out[-1]))
        return out

def get_encoder(in_channels, out_channels, num_feat, num_layers=4,bias=True):
    if num_layers == 4:
        return DC_Encoder(in_channels, out_channels, num_feat,bias=bias)
    else:
        return Encoder(in_channels, out_channels, num_feat, num_layers=num_layers,bias=bias)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims=[20], non_lin = torch.nn.ReLU, bn=False): #20
        super(MLP, self).__init__()
        layers = [nn.Linear(dim_in, hidden_dims[0])]
        for i in range(len(hidden_dims)-1):
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(non_lin())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        if bn:
            layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        layers.append(non_lin())
        layers.append(nn.Linear(hidden_dims[-1], dim_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = self.layers(x)
        # z = torch.nn.functional.normalize(z, dim=1)
        return z

class DC_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_feat, bias = True):
        """DC Encoder for images with resolution of 32x32
            Outputs of shape out_channelsx1x1

        Args:
            in_channels ([type]): [channels of input]
            out_channles ([type]): [output size]
            num_feat ([type]): [feature channels of 1st conv]
        """
        super(DC_Encoder, self).__init__()
        self.l1 = nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, bias=bias)
        self.l2 = nn.Sequential(nn.BatchNorm2d(num_feat), nn.ReLU(), nn.Conv2d(num_feat, num_feat*2, kernel_size=4, stride=2, bias=bias))
        self.l3 = nn.Sequential(nn.BatchNorm2d(num_feat*2), nn.ReLU(), nn.Conv2d(num_feat*2, num_feat*4, kernel_size=4, stride=2, bias=bias))
        self.l4 = nn.Sequential(nn.BatchNorm2d(num_feat*4), nn.ReLU(), nn.Conv2d(num_feat*4, out_channels, kernel_size=2, stride=1, bias=bias))
        self.z_dim = out_channels

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        return out

    def features(self, x):
        out = []
        out.append(self.l1(x))
        out.append(self.l2(out[-1]))
        out.append(self.l3(out[-1]))
        out.append(self.l4(out[-1]))
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_feat, bias=True ,num_layers=4):
        """DC Encoder for images with resolution of 2^(num_layers+1)
            Outputs of shape out_channelsx1x1

        Args:
            in_channels ([type]): [channels of input]
            out_channles ([type]): [output size]
            num_feat ([type]): [feature channels of 1st conv]
        """
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1, bias=bias)]
        for i in range(num_layers-2):
            layers.append(nn.Sequential(nn.BatchNorm2d(num_feat), nn.ReLU(), 
                nn.Conv2d(num_feat, num_feat*2, kernel_size=4, stride=2, padding=1, bias=bias)))
            num_feat = num_feat *2
        layers.append(nn.Sequential(nn.BatchNorm2d(num_feat), nn.ReLU(), nn.Conv2d(num_feat, out_channels, 4, 1, 0, bias=bias)))

        self.layers = nn.ModuleList(layers)
        self.z_dim = out_channels
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x)
            else:
                out = layer(out)
        return out

    def features(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = [layer(x)]
            else:
                out.append(layer(out)[-1])
        return out



class ResNet_Encoder(nn.Module):
    def __init__(self, base_model="resnet50", cifar_stem=True, channels_in=4):
        """obtains the ResNet for use as an Encoder, with the last fc layer
        exchanged for an identity

        Args:
            base_model (str, optional): [description]. Defaults to "resnet50".
            cifar_stem (bool, optional): [input resolution of 32x32]. Defaults to True.
            channels_in (int, optional): [description]. Defaults to 3.
        """
        super(ResNet_Encoder, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        self.resnet_dict = {"resnet18": ResNet18(),
                            "resnet50": ResNet50()}

        self.resnet = self._get_basemodel(base_model)
        num_ftrs = self.resnet.fc.in_features
        #num_ftrs = self.resnet.linear.in_features


        #change 1st convolution to work with inputs [channels_in x patch_x x patch_y x patch_z]
        #[channels_in x 50 x 50 x 50]
        if cifar_stem:
            conv1 = nn.Conv3d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
            self.resnet.maxpool = nn.Identity()
        elif channels_in != 3:
            conv1 = nn.Conv3d(channels_in, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
        self.resnet.fc = nn.Identity() #should it be self.resnet.linear instead? because I changed this in 239
        if base_model == 'resnet18':
            self.z_dim = 512
        else: self.z_dim = 2048



    def forward(self, x):
        return self.resnet(x)


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")




### ResNet following https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=20):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes) #self.linear
        if num_blocks[3] == 2:
            self.z_dim = 512
        else:
            self.z_dim = 2048

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) #self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



## VGG 3D following https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py#L21
'''VGG11/13/16/19 3D in Pytorch.'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels, out_channels: int = 256):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        #self.classifier = nn.Linear(512, out_channels) #10
        self.z_dim = out_channels

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        #in_channels = self._in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool3d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    # model1 = ResNet_Encoder(base_model='resnet50')
    # model2 = ResNet_Encoder(base_model='resnet18')
    # x = torch.randn(3, 3, 32, 32)
    # y1 = model1(x)
    # y2 = model2(x)
    # print(y1.shape)
    # print(y2.shape)
    model3 = ResNet_Encoder(base_model="resnet18", cifar_stem=False)
    x2 = torch.randn(1, 3, 64, 64)
    y = model3(x2)
    print(y.shape)

    net = ResNet18()
    y = net(torch.randn(1, 3, 50, 50, 50))
    print(y.size())
