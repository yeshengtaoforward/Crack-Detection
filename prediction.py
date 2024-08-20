import os
from glob import glob
import time

import pandas as pd
from PIL import ImageOps, Image
from torch.utils.data import DataLoader


from torchvision import transforms
from tqdm import tqdm

from config import valLoaderConfig, modelConfig,testLoaderConfig
from dataset_pre import CrackDataTest
# local imports

# from model.Unet_part import UNet
import  segmentation_models_pytorch as smp

from model.FCN import FCN8s, VGGNet, FCNs
from model.Model import MyModel, MyEncoder
from utils.utils import *

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
from PIL import Image
margin=5
def crop_image_by_margin(input_image_path, output_image_path, margin):
    image = Image.open(input_image_path)
    width, height = image.size
    left = margin
    upper = margin
    right = width - margin
    lower = height - margin
    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(output_image_path)
def crop_images_in_folder_by_margin(input_folder, output_folder, margin):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('_res.png', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            crop_image_by_margin(input_path, output_path, margin)

def plot_test(model, dataset, save_plots, save_path, split="test"):
    idx = 0
    if not os.path.exists(save_plots):
        os.makedirs(save_plots)
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        bar = tqdm(range(len(dataset)))
        for i in bar:
            images,path= dataset[i]
            images= images.cpu()
            # print(images.shape, masks.shape)
            images = images.unsqueeze(dim=0)
            image = transforms.ToPILImage()(images[0])

            torch.cuda.synchronize()
            start = time.time()

            mask_pred = model(images)

            torch.cuda.synchronize()
            end = time.time()

            print('infer_time:', end - start)
            masks_2 = (torch.sigmoid(mask_pred.cpu()) >= 0.5).int()
            masks_2 = masks_2.squeeze(dim=0)
            masks_2 = masks_2.to(torch.float)
            masks_2 *= 255.
            pred = transforms.ToPILImage()(masks_2.byte().cpu())
            pred = ImageOps.expand(pred, border=5, fill='white')

            (pred_width, pred_height) = pred.size

            name = path.split('/')[-1][:-4]

            final_width, final_height = (pred_width, pred_height)
            result = Image.new('RGB', (final_width, final_height))

            result.paste(im=pred)
            result.save(f"{save_plots}/{name}_FPN.png")
            bar.set_description(f"Saving Test Results")

    return

def getTestDataLoader(dfTest, **kwargs):
    dataTest = CrackDataTest(dfTest,
                             img_transforms=kwargs['test_data']['transforms'],
                             aux_transforms=None,)
    testLoader = DataLoader(dataTest,
                            batch_size=kwargs['test_data']['batch_size'],
                            shuffle=kwargs['test_data']['shuffle'],
                            pin_memory=torch.cuda.is_available(),
                            num_workers=kwargs['test_data']['num_workers'],
                            drop_last=True)

    return testLoader, dataTest

def buildModel(config, modelPath=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MyModel(encoder_name='efficientnet-b3')
    model = MyModel(encoder_name="my_encoder")
    # model=UNet(n_channels=3,n_classes=1)
    # model = smp.Unet()
    # model=smp.DeepLabV3Plus()
    # model=smp.FPN()
    # model=smp.PSPNet()
    # model=UNet(num_classes=1)
    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # model = FCNs(pretrained_net=vgg_model, n_class=1)
    model = model.to(device)
    print("loading best model...")
    model.load_state_dict(torch.load(modelPath, map_location=device))
    return model

def buildDataset(imgs_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg"))
    }
    print('数据集大小：'+str(len(data['images'])))
    df = pd.DataFrame(data)
    testLoader, dataTest = getTestDataLoader(df, test_data=testLoaderConfig)
    print(dataTest)

    return testLoader, dataTest
if __name__ == '__main__':
    #预测图片
    # image_path = r"./dataset/paper_img/CFD"
    image_path = r"./dataset/paper_img/DeepCrack"
    # image_path = r"./dataset/CRACK500/train_augmentation/imgs"

    model_path = r"save_model/logs_DeepCrack_efficientnet-b3_ECA.pth"
    plot_path = r"dataset/paper_img"
    out_path = r"dataset/paper_img"
    input_image_path = image_path
    output_image_path = image_path

    testLoader, testDataset = buildDataset(image_path)
    model = buildModel(modelConfig, model_path)

    plot_test(model, testDataset, out_path, plot_path)
    crop_images_in_folder_by_margin(input_image_path, output_image_path, margin)
