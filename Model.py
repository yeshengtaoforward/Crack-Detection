from typing import Optional, Union, List
import time
import torchsummary
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
import  segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
import math
from typing import List
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders._base import EncoderMixin

from model.FCN import VGGNet, FCNs, FCN8s
from model.deepcrack import DeepCrack
from model.layers import conv_bn_act, CoordAtt, SKConv1, CBAM, scSE_Module, GALAAttention, ECA, GlobalContextAttention
from model.layers import SamePadConv2d
from model.layers import Flatten
from model.layers import SEModule
from model.layers import DropConnect

from thop import profile
from torchstat import stat
import torchvision


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # ASPP模块中的分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[0], padding=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[1], padding=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[2], padding=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积用于特征融合
        self.conv1x1 = nn.Conv2d(out_channels*5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # ASPP模块中的分支
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 将分支特征进行拼接
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # 进行特征融合
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        # self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()
        # self.se = CoordAtt(mid_, mid_)
        self.se = SKConv1(mid_)
        # self.se = CBAM(mid_)
        # self.se = scSE_Module(mid_)
        # self.se = ECA(mid_)
        # self.se = GlobalContextAttention(mid_)


        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        # self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class MyEncoder(torch.nn.Module, EncoderMixin):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.3,
                 num_classes=1):
        super().__init__()
        self._out_channels: List[int] = [320, 112, 40, 24, 32] #b0--(320, 112, 40, 24, 32)
        #                                                         #b2--[3, 32, 24, 48, 120, 352]
                                                                  #b3--[3, 40, 32, 48, 136, 384]
        self._depth: int = 5
        self._in_channels: int = 3
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2, bias=False)

        self.blocks1 = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate)
        )
        self.blocks2 = nn.Sequential(
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
        )
        self.blocks3 = nn.Sequential(
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(5), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(5), True, 0.25, drop_connect_rate)
        )
        self.blocks4 = nn.Sequential(
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(6), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(2), True, 0.25, drop_connect_rate)
        )

        self.head = nn.Sequential(
            *conv_bn_act(renew_ch(320), renew_ch(1280), kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(renew_ch(1280), num_classes)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)
    def forward(self,x) -> List[torch.Tensor]:
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        stem = self.stem(x)
        x1 = self.blocks1(stem)
        x2 = self.blocks2(x1)
        x3 = self.blocks3(x2)
        x4 = self.blocks4(x3)
        return [x, stem, x1, x2, x3, x4]
smp.encoders.encoders["my_encoder"] = {
    "encoder": MyEncoder, # encoder class here
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://some-url.com/my-model-weights",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        # init params for encoder if any
    },
},


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, rate=1):
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels),
                      out_channels=int(out_channels),
                      kernel_size=(k_size, k_size),
                      dilation=(rate, rate),
                      padding='same'),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.convlayer(x)


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = Convolution(in_channels, in_channels, 3)
        self.conv2 = nn.Sequential(
            Convolution(in_channels, in_channels / 4, 3),
            Convolution(in_channels / 4, in_channels / 4, 1)
        )
        self.conv3 = Convolution(in_channels, in_channels / 4, 1)
        self.conv1 = nn.Sequential(
            Convolution(in_channels, in_channels / 2, 3),
            Convolution(in_channels / 2, in_channels / 2, 3, rate=2),
            Convolution(in_channels / 2, in_channels / 2, 1)
        )
        self.comb_conv = Convolution(in_channels, in_channels, 1)
        self.final = Convolution(2 * in_channels, in_channels, 3, 2)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_comb = torch.cat((x1, x2, x3), dim=1)
        x_n = self.comb_conv(x_comb)
        x_new = torch.cat((x, x_n), dim=1)
        out = self.final(x_new)
        return out

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(name=attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(name=attention_type, in_channels=out_channels)
        if skip_channels > 0:
            self.multiAttention = MultiScaleAttention(skip_channels)
        else:
            self.multiAttention = nn.Identity()
        # self.aspp=ASPP(in_channels=skip_channels,out_channels=2*skip_channels)
        # print(skip_channels)

    def forward(self, x, skip=None, i=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            # if i is not None and i == 2:
            skip = self.multiAttention(skip)
                # skip = self.aspp(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        # (320, 112, 40, 24, 32)
        # (256, 128, 64, 32, 16)

        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, i)

        return x


class MyModel(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "renet101",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] =None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        # self.encoder = MyEncoder(1, 1)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(encoder_name='efficientnet-b3')
    # model = MyModel(encoder_name="my_encoder")
    # model = smp.Unet()
    # model = smp.DeepLabV3Plus()
    # model=DeepLab(num_classes=1)
    # print(model)
    # model = smp.PSPNet()
    # model = smp.FPN()

    # FCN
    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # model = FCNs(pretrained_net=vgg_model, n_class=1)

    # model=DeepCrack()
    model = model.to(device).eval()
    # torchsummary.summary(model.cuda(), (3, 512, 512))
    # summary(model,(1,3,320,320))
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    rand_t = torch.rand((1, 3, 320, 320)).to(device)
    # out = model(rand_t)

    flops, params = profile(model, inputs=(rand_t,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
    #
    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    torch.cuda.synchronize()
    start = time.time()
    result = model(rand_t.to(device))
    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)

