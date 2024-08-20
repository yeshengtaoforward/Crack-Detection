import argparse
import os
from glob import glob
from time import time

import pandas as pd
from PIL import ImageOps, Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import  segmentation_models_pytorch as smp

from model.FCN import FCN8s, FCNs, VGGNet
from model.Model import MyModel
from model.Model import MyEncoder

from config import valLoaderConfig, modelConfig,testLoaderConfig
from dataset import CrackDataTest
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


def score_per_sample(model, dataset, criteria, save_path, split="test"):
    results_test = {"img": [], "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
                    "test_pre": [], "test_rec": [], "test_f1": []}
    model = model.cpu()
    idx = 0
    with torch.inference_mode():
        bar = tqdm(range(len(dataset)))
        for i in bar:
            images, masks, path = dataset[i]
            images, masks = images.cpu(), masks.cpu()
            images = images.unsqueeze(dim=0)

            mask_pred = model(images)
            masks_2 = (torch.sigmoid(mask_pred.cpu()) >= 0.5).int()

            loss = criteria(masks_2, masks)
            results_test['img'].append(path)
            results_test['test_loss'].append(loss.item())
            results_test['test_dice'].append(compute_dice2(masks_2, masks).item())
            results_test['test_iou'].append(get_IoU(masks_2, masks).item())
            results_test['test_acc'].append(accuracy(masks_2, masks).item())
            p, r, f = precision_recall_f1(masks_2, masks)
            results_test['test_pre'].append(p.item())
            results_test['test_rec'].append(r.item())
            results_test['test_f1'].append(f.item())
            bar.set_description(f"Saving Test Results")
            idx += 1

    data_frame = pd.DataFrame(data=results_test)
    data_frame.to_csv(f'{save_path}/results_per_image_{split}.csv')
    return


def plot_test(model, dataset, criteria, save_plots, save_path, split="test"):
    idx = 0
    if not os.path.exists(save_plots):
        os.makedirs(save_plots)

    results_test = {"img": [], "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
                    "test_pre": [], "test_rec": [], "test_f1": []}
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        bar = tqdm(range(len(dataset)))
        for i in bar:
            images, masks, path= dataset[i]
            images, masks = images.cpu(), masks.cpu()
            # print(images.shape, masks.shape)
            images = images.unsqueeze(dim=0)

            mask_pred = model(images)
            masks_2 = (torch.sigmoid(mask_pred.cpu()) >= 0.5).int()

            loss = criteria(masks_2, masks)
            results_test['img'].append(path)
            results_test['test_loss'].append(loss.item())
            results_test['test_dice'].append(compute_dice2(masks_2, masks).item())
            results_test['test_iou'].append(get_IoU(masks_2, masks).item())
            results_test['test_acc'].append(accuracy(masks_2, masks).item())
            p, r, f = precision_recall_f1(masks_2, masks)
            results_test['test_pre'].append(p.item())
            results_test['test_rec'].append(r.item())
            results_test['test_f1'].append(f.item())

            # print(images.shape, masks.shape, masks_2.shape)
            masks *= 255.
            masks_2 = masks_2.squeeze(dim=0)
            masks_2 = masks_2.to(torch.float)
            masks_2 *= 255.
            # image = transforms.ToPILImage()(images[0])
            # gt = transforms.ToPILImage()(masks.byte().cpu())
            pred = transforms.ToPILImage()(masks_2.byte().cpu())

            # image = ImageOps.expand(image, border=5, fill='white')
            # gt = ImageOps.expand(gt, border=5, fill='white')
            pred = ImageOps.expand(pred, border=5, fill='white')

            # (img_width, img_height) = image.size
            # (gt_width, gt_height) = gt.size
            (pred_width, pred_height) = pred.size

            name = path.split('/')[-1][:-4]
            # final_width, final_height = (img_width + gt_width + pred_width), max(img_height,
            #                                                                      max(gt_height, pred_height))

            final_width, final_height = (pred_width, pred_height)
            result = Image.new('RGB', (final_width, final_height))
            # result.paste(im=image, box=(0, 0))
            # result.paste(im=gt, box=(img_width, 0))
            # result.paste(im=pred, box=(img_width + gt_width, 0))
            # result.paste(im=pred)
            # result.save(f"{save_plots}/{name}_res.png")
            # bar.set_description(f"Saving Test Results")
            # if i==10:
            #     break
    data_frame = pd.DataFrame(data=results_test)
    data_frame.to_csv(f'{save_path}/logs_finished_{split}.csv')
    return


def score(model, criteria, loader):
    model.eval()
    val_logs = init_log()
    # Batch size should be 1
    bar = tqdm(loader,ncols=25)
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for idx, data in enumerate(bar):
            imgs, masks ,i= data
            imgs, masks = imgs.to(device), masks.to(device)
            output = model(imgs)
            output = output.squeeze(1)
            op_preds = torch.sigmoid(output)
            masks = masks.squeeze(1)
            loss = criteria(op_preds, masks)

            batch_size = imgs.size(0)
            val_logs['loss'].update(loss.item(), batch_size)
            val_logs['time'].update(time() - start)
            val_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
            val_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
            val_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
            p, r, f = precision_recall_f1(op_preds, masks)
            val_logs['precision'].update(p.item(), batch_size)
            val_logs['recall'].update(r.item(), batch_size)
            val_logs['f1'].update(f.item(), batch_size)

    return val_logs


def getTestDataLoader(dfTest, **kwargs):
    dataTest = CrackDataTest(dfTest,
                             img_transforms=kwargs['test_data']['transforms'],
                             mask_transform=kwargs['test_data']['transforms'],
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
    # model = MyModel(encoder_name='my_encoder')
    model = MyModel(encoder_name='efficientnet-b3')
    # model=UNet(n_channels=3,n_classes=1)
    # model=smp.Unet()
    # model=smp.DeepLabV3Plus()
    # model=smp.FPN()
    # model=smp.PSPNet()



    # model=UNet(num_classes=1)

    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # model = FCNs(pretrained_net=vgg_model, n_class=1)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)
    model = model.to(device)
    print("loading best model...")
    model.load_state_dict(torch.load(modelPath, map_location=device))
    return model


def buildDataset(imgs_path, masks_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg")),
        'masks': sorted(glob(masks_path + "/*.png"))
    }
    print('数据集大小：'+str(len(data['images'])))
    # test to see if there are images coresponding to masks
    for img_path, mask_path in zip(data['images'], data['masks']):
        # print(img_path, mask_path)
        assert img_path[-7:-4] == mask_path[-7:-4]

    df = pd.DataFrame(data)
    # print(df)
    testLoader, dataTest = getTestDataLoader(df, test_data=testLoaderConfig)

    return testLoader, dataTest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_images", type=str, help="Enter path to images folder.")
    parser.add_argument("--path_masks", type=str, help="Enter path to masks folder.")
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--result_path", type=str, help="Path to results")
    parser.add_argument("--plot_path", type=str, help="Path where plot or predictions would be saved")

    args = parser.parse_args()

    # image_path = args.path_images
    # masks_path = args.path_masks
    # model_path = args.model_path
    # plot_path = args.plot_path
    # out_path = args.result_path

    # CRACK500
    image_path = r"dataset/CRACK500/test/imgs"
    masks_path = r"dataset/CRACK500/test/masks"
    # image_path = r"dataset/CRACK500/CRACK500/test/images"
    # masks_path = r"dataset/CRACK500/CRACK500/test/masks"

    # CFD
    # image_path = f"./dataset/CFD/test_aug/image"
    # masks_path = f"./dataset/CFD/test_aug/masks"
    # CFD只做测试集
    # image_path = f"./dataset/CFD/imgs"
    # masks_path = f"./dataset/CFD/masks"

    # Deepcrack
    # image_path = f"./dataset/DeepCrack/test_img"
    # masks_path = f"./dataset/DeepCrack/test_lab"

    # image_path = f"./dataset/CrackTree260/test/images"
    # masks_path = f"./dataset/CrackTree260/test/masks"



    model_path = r"save_model/logs_CRACK500_efficientnet-b3_finnal.pth"
    plot_path = r"result/logs_CRACK500_efficientnet-b3_finnal"
    out_path = r"result/logs_CRACK500_efficientnet-b3_finnal"

    testLoader, testDataset = buildDataset(image_path, masks_path)
    model = buildModel(modelConfig, model_path)
    criteria = TverskyLoss()

    # testLog = score(model, criteria, testDataset)
    plot_test(model, testDataset, criteria, out_path, plot_path)


