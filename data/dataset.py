import os
from config import opt
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, mode=None):
        """
        目标：获取所有图像的地址，并根据训练、测试、验证划分数据
        :param root: 数据集目录
        :param transforms: 数据转换操作
        :param mode: 用来划分训练、测试和验证
        """
        assert mode in ["train", "test", "val"]

        self.mode = mode
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.mode == "test":
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        #  划分训练集、验证集，验证：训练 = 3：7
        if self.mode == "test": self.imgs = imgs
        if self.mode == "train": self.imgs = imgs[:int(0.1 * imgs_num)]
        if self.mode == "val": self.imgs = imgs[int(0.05 * imgs_num):int(0.02 * imgs_num)]

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 测试集和验证集不需要数据增强
            if self.mode == "test" or self.mode == "val":
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        # else:
        #     self.transforms = transforms

    def __getitem__(self, index):
        """
        返回一张图像的数据
        对于测试集，返回图像ID，如1000.jpg返回1000
        :param index:
        :return: 图像ID
        """
        img_path = self.imgs[index]
        if self.mode == "test":
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if "dog" in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        try:
            data = self.transforms(data)
        except RuntimeError:
            print(img_path)
            pass
        except TypeError:
            print(img_path)
            pass

        return data, label

    def __len__(self):
        """
        返回数据集中所有图像的数量
        :return:
        """
        return len(self.imgs)


if __name__ == '__main__':
    train_data = DogCat(opt.train_data_root, mode="train")

    img, label = train_data[0]
    for img, label in tqdm(train_data):
        pass
        # print((img.shape, label))

    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)

    # for ii, (data, label) in enumerate(tqdm(train_dataloader)):
    #     pass
