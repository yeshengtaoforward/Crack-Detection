from typing import Optional, Union, List
import torchsummary
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.base import modules as md
import torch
import torch.nn.functional as F
from torch import nn



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
        # self.se_aspp=SE_ASPP(dim_in=skip_channels,dim_out=skip_channels)
        # self.aspp=ASPP(in_channels=skip_channels,atrous_rates=[6,12,18],out_channels=skip_channels)

    def forward(self, x, skip=None, i=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if i is not None and i == 2:
                # skip=self.aspp(skip)
                # skip=self.se_aspp(skip)
                skip = self.multiAttention(skip)
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
            decoder_attention_type: Optional[str] = None,
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
        # self.encoder = MyEncoder(1,1)

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

    model = MyModel(encoder_name="efficientnet-b3")
    # model = smp.Unet(encoder_name="vgg19")
    # model=smp.DeepLabV3Plus()
    # model=DeepLab(num_classes=1)
    print(model)
    model = model.to(device)
    torchsummary.summary(model.cuda(), (3, 320, 640))
    # summary(model,(1,3,320,320))
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    rand_t = torch.rand((1, 3, 640, 320)).to(device)
    out = model(rand_t)
    print(out.size())
