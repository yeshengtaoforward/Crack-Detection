import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from efficientnet_pytorch import EfficientNet
from transformers import BertConfig, BertEncoder, BertIntermediate, BertLayerNorm

class EfficientNetTransformer(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNetTransformer, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)
        self.conv = nn.Conv2d(512, 256, kernel_size=1)
        self.logits = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        # Feature extraction
        feats = self.backbone.extract_features(x)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        # Transformer encoding
        feats = self.encoder(feats)

        # Decoding
        feats = feats.squeeze(1)
        feats = feats.view(-1, 32, 32, 512)
        feats = feats.permute(0, 3, 1, 2)
        feats = self.conv(feats)
        feats = self.logits(feats)
        out = F.interpolate(feats, x.size()[2:], mode='bilinear', align_corners=True)
        return out
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=EfficientNetTransformer(1)
    print(model)
    model = model.to(device)
    torchsummary.summary(model.cuda(), (3, 320, 320))
    rand_t = torch.rand((1, 3, 640, 320)).to(device)
    out = model(rand_t)
    print(out.size())