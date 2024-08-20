import torchsummary
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from time import time
from tqdm import tqdm
import gc
import argparse
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from torchsummary import summary
import segmentation_models_pytorch as smp
from model.FCN import FCN8s, VGGNet, FCNs
from model.Model import MyEncoder, MyModel
from dataset import CrackData
from model.deepcrack import DeepCrack
from utils.callbacks import CallBacks
from utils.utils import *
from utils.lrSchedular import OneCycleLR
from config import trainLoaderConfig, valLoaderConfig, modelConfig


RANDOM_STATE = 42
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




def train_step(model, optim, criteria1,loader,accumulation_steps, scaler, epoch, max_epochs):
    model.train()
    train_logs = init_log()
    bar = tqdm(loader,ncols=150)
    torch.cuda.empty_cache()
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.enable_grad():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                output = model(imgs)
                output = output.squeeze(1)
                op_preds = torch.sigmoid(output)
                masks = masks.squeeze(1)
                loss = criteria1(op_preds, masks)

                # loss = criteria(op_preds, masks) / accumulation_steps

            batch_size = imgs.size(0)

            scaler.scale(loss).backward()

            # if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(loader)):
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            train_logs['loss'].update(loss.item(), batch_size)
            train_logs['time'].update(time() - start)
            train_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
            train_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
            train_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
            p, r, f = precision_recall_f1(op_preds, masks)
            train_logs['precision'].update(p.item(), batch_size)
            train_logs['recall'].update(r.item(), batch_size)
            train_logs['f1'].update(f.item(), batch_size)

            bar.set_description(f"Training Epoch: [{epoch}/{max_epochs}] Loss: {train_logs['loss'].avg}"
                                f" Dice: {train_logs['dice'].avg} IoU: {train_logs['iou'].avg}"
                                f" Accuracy: {train_logs['acc'].avg} Precision: {train_logs['precision'].avg}"
                                f" Recall: {train_logs['recall'].avg} F1: {train_logs['f1'].avg}"
                                f"time: {train_logs['time'].avg}")
            del imgs
            del masks
            gc.collect()

    return train_logs


def val(model, criteria1, loader, accumulation_steps,epoch, epochs, split='Validation'):
    model.eval()
    val_logs = init_log()
    bar = tqdm(loader,ncols=150)
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            output = model(imgs)
            output = output.squeeze(1)
            op_preds = torch.sigmoid(output)
            masks = masks.squeeze(1)
            loss = criteria1(op_preds, masks)
            # loss = criteria(op_preds, masks)/accumulation_steps

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

            bar.set_description(f"{split} Epoch: [{epoch}/{epochs}] Loss: {val_logs['loss'].avg}"
                                f" Dice: {val_logs['dice'].avg} IoU: {val_logs['iou'].avg}"
                                f" Accuracy: {val_logs['acc'].avg} Precision: {val_logs['precision'].avg}"
                                f" Recall: {val_logs['recall'].avg} F1: {val_logs['f1'].avg}"
                                f"time: {val_logs['time'].avg}")


    return val_logs


def getDataLoaders(dfTrain, dfVal, **kwargs):
    dataTrain = CrackData(dfTrain,
                          img_transforms=kwargs['training_data']['transforms'],
                          mask_transform=kwargs['training_data']['transforms'],
                          aux_transforms=None)

    trainLoader = DataLoader(dataTrain,
                             batch_size=kwargs['training_data']['batch_size'],
                             shuffle=kwargs['training_data']['shuffle'],
                             pin_memory=torch.cuda.is_available(),
                             num_workers=kwargs['training_data']['num_workers'])

    dataVal = CrackData(dfVal,
                        img_transforms=kwargs['val_data']['transforms'],
                        mask_transform=kwargs['val_data']['transforms'],
                        aux_transforms=None)
    valLoader = DataLoader(dataVal,
                           batch_size=kwargs['val_data']['batch_size'],
                           shuffle=kwargs['val_data']['shuffle'],
                           pin_memory=torch.cuda.is_available(),
                           num_workers=kwargs['val_data']['num_workers'])

    return trainLoader, valLoader


def buildModel(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # MyModel
    model = MyModel(encoder_name='efficientnet-b3')
    # model = MyModel(encoder_name='my_encoder')

    # Unet
    # model=smp.Unet(in_channels=3,classes=1)
    # model=UNet(n_channels=3,n_classes=1)

    #Deeplabv3+
    # model=smp.DeepLabV3Plus()
    # model=DeepLab(num_classes=1,backbone="xception",pretrained=False)

    # PSPNet
    # model=smp.PSPNet()

    # FCN
    # vgg_model = VGGNet(requires_grad=True, show_params=False)
    # model = FCNs(pretrained_net=vgg_model, n_class=1)
    # model=FCN8s(pretrained_net="vgg16",n_class=1)

    # FPN
    # model=smp.FPN()
    # model=smp.MAnet()
    # model=smp.PAN
    # model=UNet(num_classes=1)
    # model=HED()

    #DeepCrack
    # model=DeepCrack()
    model = model.to(device)
    return model


def buildDataset(imgs_path, masks_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg")),
        'masks': sorted(glob(masks_path + "/*.png"))
    }
    # test to see if there are images coresponding to masks
    print('数据集大小：' + str(len(data['images'])))
    for img_path, mask_path in zip(data['images'], data['masks']):
        # print(img_path,mask_path)
        assert(img_path[-7:-4] == mask_path[-7:-4])

    df = pd.DataFrame(data)
    dfTrain, dfVal = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    print(len(dfTrain),len(dfVal))
    trainLoader, valLoader = getDataLoaders(dfTrain,
                                            dfVal,
                                            training_data=trainLoaderConfig,
                                            val_data=valLoaderConfig)

    return trainLoader, valLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_images", type=str, help="Enter path to images folder.")
    parser.add_argument("--path_masks", type=str, help="Enter path to masks folder.")
    parser.add_argument("--out_path", type=str, help="Output path, model saving path.")
    args = parser.parse_args()

    # image_path = args.path_images
    # masks_path = args.path_masks
    # out_path = args.out_path

    #CRACK500数据集
    image_path=r"./dataset/CRACK500/train_augmentation/imgs"
    masks_path =r"./dataset/CRACK500/train_augmentation/masks"
    # image_path = r"./dataset/CRACK500/train/imgs"
    # masks_path = r"./dataset/CRACK500/train/masks"
    # image_path = r"./dataset/CRACK500/CRACK500/train/images"
    # masks_path = r"./dataset/CRACK500/CRACK500/train/masks"

    # DeepCrack数据集
    # image_path = r"./dataset/DeepCrack/train_img_aug"
    # masks_path = r"./dataset/DeepCrack/train_lab_aug"
    # image_path = r"./dataset/DeepCrack/train_imgs"
    # masks_path = r"./dataset/DeepCrack/train_lab"

    # CFD
    # image_path = r"./dataset/CFD/train_aug/image"
    # masks_path = r"./dataset/CFD/train_aug/masks"

    # cracktree260
    # image_path = r"./dataset/CrackTree260/train/images"
    # masks_path = r"./dataset/CrackTree260/train/masks"

    #CDCC
    # image_path = r"./dataset/CDCC/images"
    # masks_path = r"./dataset/CDCC/masks"

    out_path = r"./save_model"


    trainLoader, valLoader = buildDataset(image_path, masks_path)

    model = buildModel(modelConfig)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(device))
    # base_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)
    # optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.06)
    schedular = OneCycleLR(optimizer, num_steps=50, lr_range=(1e-5, 0.1), annihilation_frac=0.75)

    # loss function
    criteria1 = DiceLoss()
    # criteria1 = DiceBCELoss()



    epochs = 100
    accumulation_steps = 4
    best_dice = 0.5
    scaler = GradScaler()
    out_path_model = out_path
    iteration = 0

    cb = CallBacks(best_dice, out_path_model)

    results = {"train_loss": [], "train_dice": [], "train_iou": [], 'train_acc': [],
               "train_pre": [], "train_rec": [], "train_f1": [],
               "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
               "val_pre": [], "val_rec": [], "val_f1": []}

    save_path = out_path
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # else:
    #     model_path = out_path_model
    #     if os.path.exists(model_path):
    #         model.load_state_dict(torch.load(model_path, map_location=device))

    earlyStopEpoch = 10

    try:
        start=time()
        for epoch in range(1, epochs + 1):
            iteration = epoch
            train_logs = train_step(model, optimizer, criteria1, trainLoader, accumulation_steps, scaler, epoch, epochs)
            print("\n")
            val_logs = val(model, criteria1, valLoader,accumulation_steps, epoch, epochs)
            print("\n")
            schedular.step()

            results['train_loss'].append(train_logs['loss'].avg)
            results['train_dice'].append(train_logs['dice'].avg)
            results['train_iou'].append(train_logs['iou'].avg)
            results['train_acc'].append(train_logs['acc'].avg)
            results['train_pre'].append(train_logs['precision'].avg)
            results['train_rec'].append(train_logs['recall'].avg)
            results['train_f1'].append(train_logs['f1'].avg)
            results['val_loss'].append(val_logs['loss'].avg)
            results['val_dice'].append(val_logs['dice'].avg)
            results['val_iou'].append(val_logs['iou'].avg)
            results['val_acc'].append(val_logs['acc'].avg)
            results['val_pre'].append(val_logs['precision'].avg)
            results['val_rec'].append(val_logs['recall'].avg)
            results['val_f1'].append(val_logs['f1'].avg)

            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            print(save_path)
            data_frame.to_csv(f'{save_path}/logs_CRACK500_efficientnet-b3_finnal.csv', index_label='epoch')

            print("\n")

            cb.saveBestModel(val_logs['dice'].avg, model)
            cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch)
        end=time()
        print(f' spend_time:{end-start}')

    except KeyboardInterrupt:
        data_frame = pd.DataFrame(data=results, index=range(1, iteration + 1))
        data_frame.to_csv(f'{save_path}/logs_2.csv', index_label='epoch')
        val_logs = val(model, criteria1,valLoader, 1, 1)
        cb.saveBestModel(val_logs['dice'].avg, model)
