import torch
from PIL import Image
import os
from dataprocess.dataprocess_coco import get_bbox
from torchvision import datapoints
# import torchvision.transforms.functional as FT
from torchvision.transforms.v2 import functional as F


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, label_dir, img_dir, S=7, B=2, C=20, transform=None):
        self.annotations = csv_file  # pd.read_csv(csv_file)
        self.label_dir, self.img_dir = label_dir, img_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C

    def __len__(self) : return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])# iloc[index, 1] return row: index, col: 1 = ".txt" filename
        image = Image.open(os.path.join(self.img_dir, self.annotations.iloc[index, 0]))
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x)!=int(float(x)) else int(float(x)) for x in label.strip().split()]
                boxes.append([class_label, x, y, width, height])

        if self.transform:
          image, boxes = self.transform(image, torch.tensor(boxes))

        label_matrix = torch.zeros(self.S, self.S, self.B*5+self.C)

        for box in boxes:
            class_label, x, y, width, height = box #.tolist()
            i, j = int(y*self.S), int(x*self.S)
            x_cell, y_cell, w_cell, h_cell= x*self.S-j, y*self.S-i, width*self.S, height*self.S #for x and y remove the integer part
            if label_matrix[i, j, self.C] == 0: #if no label has been assigned to this label, change it

                '''
                 #having only 1 class = '0', we will always have the first 2 elems = 1
                 l_m[0] => 1 (relative to the correct class)
                 l_m[1] => 1 (always 1, relative to the number of classes, relative to the existence of an object)

                 #example with 2 classes, and the first(class 0) is the correct one
                 l_m[0] => 1 (relative to the correct class)
                 l_m[1] => 0 (relative to the wrong)
                 l_m[2] => 1 (always 1, relative to the number of classes, relative to the existence of an object)

                 then we have always 8 elements relative to the bounding boxes elements

                '''

                '''

                IMPORTANT: this model can predict at most 1 box for cell, in fact is useless to have (B*5+C) it should be enough (5+C):
                1: to tell the model there is an object
                4: to specify (x, y, w, h)
                C: set to 1 the correct class
                '''

                label_matrix[i, j, self.C], label_matrix[i, j, int(class_label)] = 1, 1
                label_matrix[i, j, self.C+1:self.C+5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])

        return image,label_matrix
      
class VOCDatasetPredict(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, S=7, B=2, C=20, transform=None):
        self.annotations = csv_file  # pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C

    def __len__(self) : return len(self.annotations)

    def __getitem__(self, index):
        
        image = Image.open(os.path.join(self.img_dir, self.annotations.iloc[index, 0]))

        if self.transform:
          image = self.transform(image)

        return image
      
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data, image_folder, S=7, B=2, C=20, transform=None,filter_category=None):
        #self.image_id_list = image_id_list  # pd.read_csv(csv_file)
        self.data = data
        self.image_list=list(sorted(os.listdir(image_folder)))
        self.img_dir=image_folder
        self.transform = transform
        self.filter_category=filter_category
        self.S, self.B, self.C = S, B, C

    def __len__(self) : return len(self.image_list)

    def __getitem__(self, index):
        if self.image_list[index][len(self.image_list[index])-4:] != '.jpg':
          index=index+1
        image_id=int(self.image_list[index][:len(self.image_list[index])-4])
        # image_id_str_=str(image_id)
        # image_id_pad=image_id_str_.zfill(12)
        # url='http://images.cocodataset.org/train2017/'+image_id_pad+'.jpg'
        # image_raw= requests.get(url, stream=True).raw
        # image = Image.open(image_raw).convert('RGB')
        image = Image.open(os.path.join(self.img_dir, self.image_list[index]))
        boxes = get_bbox(self.data, image_id,self.filter_category)

        #print(image,boxes)

        if self.transform: image, boxes = self.transform(image, torch.tensor(boxes))
        label_matrix = torch.zeros(self.S, self.S, self.B*5+self.C)
        for box in boxes:
            class_label, x, y, width, height = box#.tolist()
            i, j = int(y*self.S), int(x*self.S)
            x_cell, y_cell, w_cell, h_cell= x*self.S-j, y*self.S-i, width*self.S, height*self.S
            if label_matrix[i, j, self.C] == 0 :
                label_matrix[i, j, self.C], label_matrix[i, j, int(class_label)] = 1, 1
                label_matrix[i, j, self.C+1:self.C+5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])

        return image,label_matrix
      
def yolotobox(boxes,dw,dh):
    new_boxes=[]
    for box in boxes:
      box=box[len(box)-4:]
      x,y,w,h=box[0],box[1],box[2],box[3]
      cx=x*dw
      cy=y*dh
      w1=w*dw
      h1=h*dh

      if cx < 0: cx = 0
      if cy < 0: cy = 0
      if w1 > dw - 1: w1 = dw - 1
      if h1 > dh - 1: h1 = dh - 1
      new_boxes.append([cx, cy, w1, h1])
    return new_boxes

def boxtoyolo(boxes,dw,dh):
    boxes=boxes.tolist()
    new_boxes=[]
    for box in boxes:
      x,y,w,h=box[0],box[1],box[2],box[3]
      cx=x/dw
      cy=y/dh
      w1=w/dw
      h1=h/dh
      if w1<=0.05 or h1<=0.05:
        continue
      new_boxes.append([0.0,cx, cy, w1, h1])
    return new_boxes
  
class ComposeDataAugmentation(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        bboxes=yolotobox(bboxes,F.get_spatial_size(img)[1],F.get_spatial_size(img)[0])
        if not bboxes:
          bboxes.append([0.0,0.0,0.0,0.0])
        bboxes = datapoints.BoundingBox(bboxes,format=datapoints.BoundingBoxFormat.CXCYWH,spatial_size=F.get_spatial_size(img),)
        for t in self.transforms:
            img, bboxes = t(img,bboxes)

        return img, boxtoyolo(bboxes,448,448)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

      
class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, annotation_path, img_dir, img_size, name2idx):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx
        
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
        
    def __len__(self):
        return self.img_data_all.size(dim=0)
    
    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]
        
    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        
        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        
        for i, img_path in enumerate(img_paths):
            
            # skip if the image path is not valid
            if (not img_path) or (not os.path.exists(img_path)):
                continue
                
            # read and resize image
            img = io.imread(img_path)
            img = resize(img, self.img_size)
            
            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            
            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])
            
            img_data_all.append(img_tensor)
            gt_idxs_all.append(gt_idx)
        
        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad