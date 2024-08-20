from torchvision import transforms

trainLoaderConfig = {
    'transforms': transforms.Compose([transforms.Resize((320, 320)),
                                      # transforms.Pad(padding=(0, 32), fill=0, padding_mode='constant'),
                                      # transforms.RandomHorizontalFlip(p=0.5),
                                      # transforms.RandomVerticalFlip(p=0.25),
                                      transforms.ToTensor(),
    ]),
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 0
}

valLoaderConfig = {
    'transforms': transforms.Compose([transforms.Resize((320, 320)),
                                      transforms.ToTensor(),
                                      ]),
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 0
}
testLoaderConfig = {
    'transforms': transforms.Compose([transforms.Resize((320, 320)),
                                      transforms.ToTensor(),
                                      ]),
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 0
}
modelConfig = {
    'encoderBackbone1': 'efficientnet-b2',
    'encoderBackbone2': 'resnet34',
    'encoderBackbone3':'vgg16',
    'encoderBackbone4':'mobilenet_v2',
    'encoderBackbone5': 'resnet18',
    'encoderBackbone6': 'timm-mobilenetv3_large_100'
}
