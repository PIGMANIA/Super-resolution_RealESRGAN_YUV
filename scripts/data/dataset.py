import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy

from data.data_prepare import DataPrepare
from utils import check_image_file


class Dataset(Dataset):
    def __init__(self, cfg):

        self.data_pipeline = DataPrepare(cfg)

        self.hrfiles = [
            os.path.join(cfg.train.dataset.train_dir, x)
            for x in os.listdir(cfg.train.dataset.train_dir)
            if check_image_file(x)
        ]

        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Read input data and output data
        hr = Image.open(self.hrfiles[index]).convert("RGB")
        lr, hr = self.data_pipeline.data_pipeline(hr)
        
        lr = Image.fromarray(lr).convert("YCbCr")
        hr = Image.fromarray(hr).convert("YCbCr")

        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len
