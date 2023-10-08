import torch.nn as nn
from torchvision.models import resnet50
from ssd.modeling import registry
from ssd.modeling.backbone.vgg import extras_base
from ssd.utils.model_zoo import load_state_dict_from_url
import torch.nn.functional as F
model_urls = {
    'resnet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    img_size = size  # assume square image
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if img_size // 2 < 3:  # Check if the feature size would become too small
                    break
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
                img_size = img_size // 2
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 512, kernel_size=1, stride=1))  # 添加一个2048到512的1x1卷积层
        layers.append(nn.Conv2d(512, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
    return layers


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        resnet = resnet50(pretrained=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 注意这里修改了 extras1 的输入通道数
        extras_config = extras_base[str(size)]
        self.extras1 = nn.ModuleList(add_extras(extras_config, i=256, size=size))  # i=256
        self.extras2 = nn.ModuleList(add_extras(extras_config, i=512, size=size))  # i=512
        self.extras3 = nn.ModuleList(add_extras(extras_config, i=1024, size=size))  # i=1024
        self.reset_parameters()


    def reset_parameters(self):
        for m in self.extras1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.extras2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.extras3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        features = [x]
        for k, v in enumerate(self.extras1):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        x = self.layer3(x)
        for k, v in enumerate(self.extras2):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        x = self.layer4(x)
        for k, v in enumerate(self.extras3):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)

@registry.BACKBONES.register('resnet')
def resnet(cfg, pretrained=True):
    model = ResNet(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['resnet']))
    return model