import torch
import os
from hyperparams import hypers

def predict_yolo_v8(model, test_imgs_dir, prediction_dir):
  with torch.no_grad():
    results=model.predict(source=test_imgs_dir,conf=0.50,iou=0.75)

  separator = "\\" if hypers.DATASET_PATH == "LOCAL" else "/"
  test_img_list=[]
  for result in results:
      if len(result.boxes.xyxy):
          
          name=result.path.split(separator)[-1].split(".")[0]
          boxes=result.boxes.xyxy.cpu().numpy()
          scores=result.boxes.conf.cpu().numpy()

          test_img_list.append(name)

          label_file_path=os.path.join(prediction_dir,name+".txt")
          # print(prediction_dir, name)
          with open(label_file_path,"w+") as f:
              for score,box in zip(scores,boxes):
                  text=f"{score:0.4f} "+" ".join(box.astype(str))
                  f.write(text)
                  f.write("\n")
  
  return (results, test_img_list)