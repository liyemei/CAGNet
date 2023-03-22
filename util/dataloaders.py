import os
import random
from PIL import Image
import torch
import torch.utils.data as data

from util import transforms as tr

def get_loaders(opt):

    train_dataset = CDDloader(opt, 'train', aug=True)
    val_dataset = CDDloader(opt, 'val', aug=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_eval_loaders(opt):
    dataset_name = "test"
    print("using dataset: {} set".format(dataset_name))
    eval_dataset = CDDloader(opt, dataset_name, aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers)
    return eval_loader

class CDDloader(data.Dataset):
    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.dual_label = opt.dual_label
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(os.path.join(self.data_dir, phase, 'A'))]
        names.sort()
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)

    def __getitem__(self, index):
        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 'A', name))
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 'B', name))
        label_name = name.replace("tif", "png") if name.endswith("tif") else name   # for shengteng
        label1 = Image.open(os.path.join(self.data_dir, self.phase,'OUT', label_name))

        if self.dual_label:
            label2 = Image.open(os.path.join(self.data_dir, self.phase, 'label2', label_name))
        else:
            label2 = label1
        if self.aug:
            img1, img2, label1, label2 = tr.with_augment_transforms([img1, img2, label1, label2])
        else:
            img1, img2, label1, label2 = tr.without_augment_transforms([img1, img2, label1, label2])

        return img1, img2, label1, label2, name

    def __len__(self):
        return len(self.names)

def is_img(name):
    img_format = ["jpg", "png", "jpeg", "bmp", "tif", "tiff", "TIF", "TIFF"]
    if "." not in name:
        return False
    if name.split(".")[-1] in img_format:
        return True
    else:
        return False

