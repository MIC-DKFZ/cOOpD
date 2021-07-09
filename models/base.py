import torch
import torch.nn as nn
import torchvision.models as models

def get_encoder(in_channels, out_channels, num_feat, num_layers=4,bias=True):
    if num_layers == 4:
        return DC_Encoder(in_channels, out_channels, num_feat,bias=bias)
    else:
        return Encoder(in_channels, out_channels, num_feat, num_layers=num_layers,bias=bias)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims=[20], non_lin = torch.nn.ReLU, bn=False):
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
    def __init__(self, base_model="resnet50", cifar_stem=True, channels_in=3):
        """obtains the ResNet for use as an Encoder, with the last fc layer
        exchanged for an identity

        Args:
            base_model (str, optional): [description]. Defaults to "resnet50".
            cifar_stem (bool, optional): [input resolution of 32x32]. Defaults to True.
            channels_in (int, optional): [description]. Defaults to 3.
        """
        super(ResNet_Encoder, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        self.resnet = self._get_basemodel(base_model)
        num_ftrs = self.resnet.fc.in_features

        #change 1st convolution to work with inputs [channels_in x 32 x 32]
        if cifar_stem:
            conv1 = nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
            self.resnet.maxpool = nn.Identity()
        elif channels_in != 3:
            conv1 = nn.Conv2d(channels_in, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
        self.resnet.fc = nn.Identity()
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