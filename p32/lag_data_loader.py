import os
import random
import torch.utils.data

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend

    return pil_loader(path)


class LAGDataLoader(torch.utils.data.Dataset):
    # constructor of the class
    def __init__(self, path, transform, shuffle=False):
        self.transform = transform

        normal_path   = os.path.join(path, 'normal_test')
        abnormal_path = os.path.join(path, 'abnormal_test')
        gt_path       = os.path.join(path, 'attention_map')
        self.normal_img    = sorted([os.path.join(normal_path, img) for img in os.listdir(normal_path)])
        self.abnormal_img  = sorted([os.path.join(abnormal_path, img) for img in os.listdir(abnormal_path)])
        self.images = self.normal_img + self.abnormal_img
        # self.gt_img        = sorted([os.path.join(path, img) for img in os.listdir(gt_path)])

    def __getitem__(self, index):
        # norm_data, norm_label = self.data_process(index, self.abnormal_img)
        image_path = self.images[index]
        if image_path.split('/')[-2] == 'normal_test':
            label = 0
        elif image_path.split('/')[-2] == 'abnormal_test':
            label = 1
        else:
            Exception("check test files")
        data = default_loader(image_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)


class DualDataLoader(torch.utils.data.Dataset):

    # constructor of the class
    def __init__(self, path1, path2, transform):
        self.transform = transform
        images_1 = [os.path.join(path1, img) for img in os.listdir(path1)]
        images_1 = sorted(images_1)
        images_2 = [os.path.join(path2, img) for img in os.listdir(path2)]
        images_2 = sorted(images_2)
        self.images_1 = images_1
        self.images_2 = images_2

    def __getitem__(self, index):
        # 1
        image_path1 = self.images_1[index]
        label1 = image_path1.split('/')[-1].split('.')[0]
        data1 = default_loader(image_path1)
        data1 = self.transform(data1)

        # # 2
        image_path2 = self.images_2[index]
        label2 = image_path2.split('/')[-1].split('.')[0]
        data2 = default_loader(image_path2)
        data2 = self.transform(data2)

        return data1, label1, data2, label2

    def __len__(self):
        return len(self.images_1)
