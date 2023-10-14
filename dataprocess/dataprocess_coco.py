import json
import cv2
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import shutil
#import zipfile module
from zipfile import ZipFile
from PIL import Image
import requests
from os import path
from tqdm import tqdm




def get_img_ann(data, image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None
      
def get_single_category(data, category_number):
  images=[]
  for annot in data['annotations']:
      if annot['category_id']==category_number: #car = 3
        images.append(annot['image_id'])
  return images


def get_img(data, image_id):
  for img in data['images']:
    if img['id'] == image_id:
      return img
    
def get_bbox(data, image_id, filter_cat=None):

    img = get_img(data, image_id)
    img_id = img['id']
    img_w = img['width']
    img_h = img['height']

    img_ann = get_img_ann(data, img_id)

    box_list=[]
    for ann in img_ann:
      current_category = ann['category_id'] - 1 # As yolo format labels start from 0
      if filter_cat is not None:
        if current_category not in filter_cat:
            continue
      current_bbox = ann['bbox']
      x = current_bbox[0]
      y = current_bbox[1]
      w = current_bbox[2]
      h = current_bbox[3]

      # Finding midpoints
      x_centre = (x + (x+w))/2
      y_centre = (y + (y+h))/2

      # Normalization
      x_centre = x_centre / img_w
      y_centre = y_centre / img_h
      w = w / img_w
      h = h / img_h

      # Limiting upto fix number of decimal places
      x_centre = float(format(x_centre, '.6f'))
      y_centre = float(format(y_centre, '.6f'))
      w = float(format(w, '.6f'))
      h = float(format(h, '.6f'))
      box_list.append([current_category,x_centre,y_centre,w,h])
      
    return box_list

def download_coco_images(car_images, number_images, coco_dir):
  for image_id in tqdm(car_images[:number_images], position=0, leave=True):
    if not os.path.exists(coco_dir+"/"+str(image_id)+".jpg"):
      image_id_str_=str(image_id)
      image_id_pad=image_id_str_.zfill(12)
      url='http://images.cocodataset.org/train2017/'+image_id_pad+'.jpg'
      image_raw= requests.get(url, stream=True).raw
      image = Image.open(image_raw).convert('RGB')
      image.save(coco_dir+"/"+str(image_id)+".jpg")
      

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width, box[3] * height,
            linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()

def plot_real_images(data_coco, coco_dir, end_index=-1):
  for index, image in enumerate(list(sorted(os.listdir(coco_dir)))):
    # print(image)
    bbox=get_bbox(data_coco, int(image[:len(image)-4]),[2])
    # print(image,bbox)
    b=[]
    for box in bbox:
        box.insert(0,0)
        b.append(box)
    # print(b)
    plot_image(Image.open(coco_dir+'/'+image),b)
    if index == end_index:
      return
      
def plot_predicted_images(coco_dir_test, result):
  for key in result:
    if len(result[key])>0:
      print(key,result[key])
      img_file_path=os.path.join(coco_dir_test,key)
      img = Image.open(img_file_path)
      print(img_file_path)
      plot_image(img, result[key])
      