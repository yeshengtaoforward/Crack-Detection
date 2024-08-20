from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image
import shutil

def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-2, 2)*90
    if random_angle==0:
     rotation_img = img.rotate(-90) #旋转角度
    else:
        rotation_img = img.rotate( random_angle)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def createImage(imageDir,saveDir):
   i=0
   for name in os.listdir(imageDir):
      i=i+1
      # 对比度增强
      # saveName="cesun"+name
      # saveImage=contrastEnhancement(imageDir,name)
      # saveImage.save(os.path.join(saveDir,saveName))

      # 翻转
      saveName1 ="flip_"+name
      saveImage1 = flip(imageDir,name)
      saveImage1.save(os.path.join(saveDir, saveName1))

      # 亮度
      # saveName2 = "brightnessE" + name
      # saveImage2 = brightnessEnhancement(imageDir, name)
      # saveImage2.save(os.path.join(saveDir, saveName2))
      print(saveName1+'已保存')

import os.path
import shutil
num = 0 #修改文件名的数量词

#保存图片模块
def moveFiles(path, disdir):  # path为原始路径，disdir是移动的目标目录

    dirlist = os.listdir(path)
    for i in dirlist:
        print(i)
        child = os.path.join('%s/%s' % (path, i))
        if os.path.isfile(child):
            imagename, jpg = os.path.splitext(i)  # 分开文件名和后缀
            imagename="brightnessE"+imagename
            shutil.copy(child, os.path.join(disdir, imagename + ".png"))#保存格式自己设置



if __name__ == '__main__':
    imageDir="dataset/Crack500/train/imgs" #要改变的图片的路径文件夹
    saveDir=r"dataset/Crack500/train_augmentation_2/imgs"   #数据增强生成图片的路径文件夹

    # imageDir =r"dataset/Deepcrack/train_lab"  # 要改变的图片的路径文件夹
    # saveDir = r"dataset/Deepcrack/train_lab_aug"  # 数据增强生成图片的路径文件夹

    # imageDir=r"dataset/CFD/test/masks" #要改变的图片的路径文件夹
    # saveDir=r"dataset/CFD/test_aug/masks"   #数据增强生成图片的路径文件夹
    # for name in os.listdir(imageDir):
    #     print(name)
    createImage(imageDir,saveDir)
    # moveFiles(imageDir, saveDir)
