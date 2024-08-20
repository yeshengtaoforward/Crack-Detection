import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, target_size)

def resize_image(input_image_path, output_image_path, target_size):
    image = Image.open(input_image_path)
    resized_image = image.resize(target_size, Image.LANCZOS)
    resized_image.save(output_image_path)

# 示例用法
input_folder = r'D:\yst\DeepCrack-master\codes\data\paper_img_deepcrack'
output_folder = r'D:\yst\DeepCrack-master\codes\data\paper_img_deepcrack_resize'
target_size = (384, 544)  # 目标尺寸 (宽度, 高度)

resize_images_in_folder(input_folder, output_folder, target_size)
