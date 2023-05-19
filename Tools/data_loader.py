# load data and split data
import os
import torch
import torch.utils.data as data
from torchvision import datasets
from PIL import Image
from Tools.utils import test_trans
from torch.utils.data.dataset import Dataset

# load image
def load_image(path, trans):
    image = Image.open(path)
    return trans(image)

# load dataset
class Folders(Dataset):
    def __init__(self, base_dir, transform):
        super().__init__()
        self.data = []
        clses = os.listdir(base_dir)
        for x in clses:
            x_dir = os.path.join(base_dir, x)
            files = os.listdir(x_dir)
            for f in files:
                self.data.append([f, x])
        self.base_dir = base_dir
        self.transform = transform

    def __getitem__(self, index):
        f,x = self.data[index]
        x_dir = os.path.join(self.base_dir, x)
        image_path = os.path.join(x_dir, f)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, x, f

    def __len__(self):
        return len(self.data)

# automatically split data into training data and testing data if ratio is given
def loadNsplit(path, ratio, image_size):
    # ratio: train/total
    assert 0<ratio<1, "ratio must be between 0 and 1"
    # load data
    train_datasets = datasets.ImageFolder(path, transform=test_trans(image_size))
    train_size = int(ratio * len(train_datasets))
    test_size = len(train_datasets) - train_size
    # random_split() random split dataset into non-overlapping new datasets of given lengths
    train_set, test_set = data.random_split(train_datasets, [train_size, test_size], 
                                                    generator=torch.Generator().manual_seed(1234))
    # return train_set, test_set
    return train_set, test_set
