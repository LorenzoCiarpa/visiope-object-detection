import torch
import torch.nn as nn
import os
from dataprocess.dataprocess import *
from PIL import Image
import torchvision.transforms as transforms
from hyperparams import hypers

architecture_config_v1 = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

      
architecture_config_v2 = [
    (3, 16, 1, 1),
    "M-2",
    # "BN",
    # "D",
    (3, 32, 1, 1),
    "M-2",
    # "BN",
    (3, 64, 1, 1),
    "M-2",
    # "BN",
    # "D",
    (3, 128, 1, 1),
    "M-2",
    # "BN",
    (3, 256, 1, 1),
    "M-2",
    # "D",
    # "BN",
    (3, 512, 1, 1),
    "M-2",
    # "BN",
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    (1, 1024, 1, 0)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) # not present in the original YOLO model
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x): return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config_v1
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # print(f'output shape: {x.shape}')
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        in_channels = self.in_channels
        layers = []
        for x in architecture:
            if type(x)==tuple:
                layers += [ CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]) ]
                in_channels = x[1]
            elif type(x)==str:
                layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]
            elif type(x)==list:
                conv1, conv2, num_repeats = x[0], x[1], x[2]
                for i in range(num_repeats):
                    layers += [ CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]) ]
                    layers += [ CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]) ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S,496), #4096 in YOLO paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(B*5+C))
        )
    def predict(self, source,conf=0.50,iou=0.75):
        self.eval()
        with torch.no_grad():

          test_img_list=list(sorted(os.listdir(source)))
          result={}
          for image in test_img_list:
            img_file_path=os.path.join(source,image)
            img = Image.open(img_file_path)
            for t in [transforms.Resize((448, 448)), transforms.ToTensor(),]:
              img = t(img)
            img=torch.unsqueeze(img, dim=0)
            img = img.to(hypers.DEVICE)
            bboxes = cellboxes_to_boxes(self(img),S=7,B=2, C=1)
            bboxes = non_max_suppression(bboxes[0], iou_threshold=iou, threshold=conf, box_format="midpoint")
            result[image]=bboxes
        return result

class Yolov2Tiny(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config_v2
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # print(f'output shape: {x.shape}')
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        in_channels = self.in_channels
        layers = []
        for x in architecture:
            if type(x)==tuple:
                layers += [ CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]) ]
                in_channels = x[1]
            elif type(x)==str:
                layer_type = x.split("-")
        
                if layer_type[0] == "M":
                  stride = int(layer_type[1])
                  layers += [ nn.MaxPool2d(kernel_size=2, stride=stride) ]
                elif layer_type[0] == "BN":
                  layers += [ nn.BatchNorm2d(in_channels) ]
                elif layer_type[0] == "D":
                  layers += [ nn.Dropout2d(p=0.2) ]
                  
            elif type(x)==list:
                conv1, conv2, num_repeats = x[0], x[1], x[2]
                for i in range(num_repeats):
                    layers += [ CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]) ]
                    layers += [ CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]) ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S,496), #4096 in YOLO paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(B*5+C))
        )
      
    def predict(self, source,conf=0.50,iou=0.75):
        self.eval()
        with torch.no_grad():

          test_img_list=list(sorted(os.listdir(source)))
          result={}
          for image in test_img_list:
            img_file_path=os.path.join(source,image)
            img = Image.open(img_file_path)
            for t in [transforms.Resize((448, 448)), transforms.ToTensor(),]:
              img = t(img)
            img=torch.unsqueeze(img, dim=0)
            img = img.to(hypers.DEVICE)
            bboxes = cellboxes_to_boxes(self(img),S=7,B=2, C=1)
            bboxes = non_max_suppression(bboxes[0], iou_threshold=iou, threshold=conf, box_format="midpoint")
            result[image]=bboxes
        return result