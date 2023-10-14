import torch
from PIL import Image
import os
from dataprocess.dataprocess_coco import get_bbox

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
      
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
