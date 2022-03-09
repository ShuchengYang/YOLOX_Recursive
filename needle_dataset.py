import json
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data
from torch.utils.data import DataLoader

class NeedleDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.input_transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def parse_my_label(self, label_json):
        x1 = label_json['labels'][0]['x1']
        x2 = label_json['labels'][0]['x2']
        y1 = label_json['labels'][0]['y1']
        y2 = label_json['labels'][0]['y2']

        xlim = label_json['labels'][0]['size']['width']
        ylim = label_json['labels'][0]['size']['height']

        xmin = max(0, x1)
        xmax = min(x2, xlim)
        ymin = max(0, y1)
        ymax = min(y2, ylim)

        xmin = 640. * xmin / xlim
        xmax = 640. * xmax / xlim
        ymin = 640. * ymin / ylim
        ymax = 640. * ymax / ylim

        xc = (xmin + xmax) / 2.
        yc = (ymin + ymax) / 2.
        bw = xmax - xmin
        bh = ymax - ymin
        # 在new head 里面前四位是坐标，后一位是种类
        label = torch.tensor([xc, yc, bw, bh, 0]).unsqueeze(0)
        return label

    def __getitem__(self, idx):
        img_path = self.img_dir+'/'+str(idx)+'.png'
        ann_path = self.ann_dir+'/'+str(idx)+'.json'

        image = cv2.imread(img_path)
        origin_imgh, origin_imgw, _ = image.shape
        image = transforms.ToTensor()(cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC))

        label_file = open(ann_path)
        label_json = json.load(label_file)
        label = self.parse_my_label(label_json)
        label_file.close()

        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class NeedleData():
    def __init__(self, img_folder_path, ann_folder_path, batch, shuffle):
        super().__init__()
        self.img = img_folder_path
        self.ann = ann_folder_path
        self.batch = batch
        self.shuffle = shuffle

    def get_dataset(self):
        return NeedleDataset(self.img, self.ann)

    def get_dataloader(self):
        dataset = self.get_dataset()
        return DataLoader(dataset,batch_size=self.batch, shuffle=self.shuffle)

    def get_datalen(self):
        dataset = self.get_dataset()
        return len(dataset)