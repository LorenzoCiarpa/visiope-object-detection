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