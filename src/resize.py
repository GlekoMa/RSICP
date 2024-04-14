# Date: 2024/4/13
import os
from torchvision import transforms
from torchvision.io import read_image, write_png


def resize_image_to_800(ori_path, tar_path):
    """
    Resize an image to 800x800 pixels. If the original image's width is greater
    than its height, resize the height to 800 while maintaining the original 
    width-to-height ratio. Then randomly crop an 800x800 window as the target 
    image. The same applies vice versa. e.g. 10000x8000 => 1000x800 => 800x800.
    """
    trans = transforms.Compose(
        [transforms.Resize(size=800), transforms.RandomCrop(size=800)]
    )
    image = read_image(ori_path)
    image_resized = trans(image)
    write_png(image_resized, tar_path)


if __name__ == "__main__":
    ori_root = "../data/Chinese-Painting/images"
    tar_root = "../data/Chinese-Painting-s800/images"
    os.makedirs(tar_root)
    image_names = os.listdir(ori_root)
    ori_paths = [os.path.join(ori_root, name) for name in image_names]
    tar_paths = [os.path.join(tar_root, name) for name in image_names]
    for ori, tar in zip(ori_paths, tar_paths):
        resize_image_to_800(ori, tar)
