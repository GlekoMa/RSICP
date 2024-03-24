import os
from torchvision import transforms
from torchvision.io import read_image, write_png

def resize_image_to_800(ori_path, tar_path):
    trans = transforms.Compose([
        transforms.Resize(size=800),
        transforms.RandomCrop(size=800)
    ])
    image = read_image(ori_path)
    image_resized = trans(image)
    write_png(image_resized, tar_path)


ori_root = "../data/RSICP_mini"
tar_root = "./images"

image_names = os.listdir(ori_root)

ori_paths = [os.path.join(ori_root, name) for name in image_names]
tar_paths = [os.path.join(tar_root, name) for name in image_names]

for ori, tar in zip(ori_paths, tar_paths):
    resize_image_to_800(ori, tar)
