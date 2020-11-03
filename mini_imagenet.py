import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.misc
import numpy as np

ROOT_PATH = './materials/'
ROOT_PATH_IMG = './materials/images/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH_IMG,  name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        permutation = np.random.permutation(len(label))

        self.data = np.array(data)[permutation]
        self.label = np.array(label)[permutation]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        # image = self.transform(Image.open(path).convert('RGB'))
        image = scipy.misc.imread(path, mode='RGB')

        # Infer class from filename.


        # Central square crop of size equal to the image's smallest side.
        height, width, channels = image.shape
        crop_size = min(height, width)
        start_height = (height // 2) - (crop_size // 2)
        start_width = (width // 2) - (crop_size // 2)
        image = image[
                start_height: start_height + crop_size,
                start_width: start_width + crop_size, :]

        image = scipy.misc.imresize(image, (84, 84), interp='bilinear')
        image = self.transform(image)

        return image, label
