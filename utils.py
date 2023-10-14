import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataprocess.dataprocess import *


def save_sample_pics(test_imgs_dir, imgs_name=None):
  fig,axes=plt.subplots(1,1,figsize=(12,12))
  plt.subplots_adjust(wspace=0.1,hspace=0.1)
  #ax=axes.flatten()
  if imgs_name is None:
    imgs_name=np.random.choice(test_img_list,15)
  
  for i,img_name in enumerate(imgs_name):
      img_file_path=os.path.join(test_imgs_dir,img_name+".jpg")
      img=cv2.imread(img_file_path)
      img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

      label_file_path=os.path.join(prediction_dir,img_name+".txt")
      label=pd.read_csv(label_file_path,sep=" ",header=None).values
      scores=label[:,0]
      boxes=label[:,1:]
      show_bbox_v8(img,boxes,scores,name=img_name+"_yolov8")
      
  plt.savefig("1.png")
  
def makeCSVfile(dataset, folder):
    images = subprocess.check_output("ls "+ dataset.location + folder + '/images', shell=True, text=True).split('\n')[:-1]
    labels = subprocess.check_output("ls "+ dataset.location + folder + '/labels', shell=True, text=True).split('\n')[:-1]
    return pd.DataFrame(data = {'image': images, 'text':labels})

def save_checkpoint(state, filename="my_checkpoint.pth.tar", exit_training=False):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("=> Saved checkpoint")
    if exit_training : exit()

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
#TO SHOW THE BOXES PREDICTED
# index = 0

# for x, y in train_loader:
#     x = x.to(DEVICE)
#     index += 1
#     for idx in range(8):
#         if index < 2:
#             continue
#         bboxes = cellboxes_to_boxes(model(x), S=7, B=2, C=1)
#         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
#         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

#     # import sys
#     # sys.exit()

#     if index == 3:
#         break