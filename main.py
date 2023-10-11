import torch

import hypers
from models.yolo import Yolov2Tiny
from models.loss import YoloLossMultiBoxes
from train import train_fn
from dataprocess.dataset import VOCDataset
from dataprocess.dataprocess import create_annotations, get_bboxes, mean_average_precision



DATASET_PATH = "LOCAL"
if dataset_path == "LOCAL":
    drive_dir=os.path.join(os.getcwd(), "archive","data")
elif dataset_path == "DRIVE":
    drive_dir=os.path.join(os.getcwd(),"drive","MyDrive","archive","data")
    
    


#Dataframe Creation
train_labels=os.path.join(drive_dir,"train_solution_bounding_boxes (1).csv")

train_imgs_dir=os.path.join(drive_dir,"training_images")
test_imgs_dir=os.path.join(drive_dir,"testing_images")

imgs_list=list(sorted(os.listdir(train_imgs_dir)))
idxs=list(range(len(imgs_list)))
np.random.shuffle(idxs)

train_idx=idxs[:int(0.8*len(idxs))]
val_idx=idxs[int(0.8*len(idxs)):]


imgs_list=list(sorted(os.listdir(train_imgs_dir)))

model = Yolov2Tiny(split_size=7, num_boxes=hypers.NUM_BOXES, num_classes=hypers.NUM_CLASSES).to(hypers.DEVICE)
optimizer = optim.Adam( model.parameters(), lr=hypers.LEARNING_RATE, weight_decay=hypers.WEIGHT_DECAY )
loss_fn = YoloLossMultiBoxes(S=7, B=hypers.NUM_BOXES, C=hypers.NUM_CLASSES)

if hypers.LOAD_MODEL:
    load_checkpoint(torch.load(hypers.LOAD_MODEL_FILE), model, optimizer)

annotations=create_annotations(images_dir) # LIST LIKE: 0     vid_4_1000.jpg   vid_4_1000.txt

train_dataset=VOCDataset(annotations, labels_dir, images_dir, S=7, B=hypers.NUM_BOXES, C=hypers.NUM_CLASSES, transform=transform)


    #train_dataset = VOCDataset( makeCSVfile(dataset, '/train'),
    #                           transform=transform, img_dir=IMG_DIR_TRAIN, label_dir=LABEL_DIR_TRAIN)

    #test_dataset = VOCDataset( makeCSVfile(dataset, '/test'),
    #                          transform=transform, img_dir=IMG_DIR_TEST, label_dir=LABEL_DIR_TEST)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hypers.BATCH_SIZE,
                              pin_memory=hypers.PIN_MEMORY, shuffle=True, drop_last=True)

    #test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    #                         pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    for epoch in range(hypers.EPOCHS):
        #for x, y in train_loader:
        #    x = x.to(DEVICE)
        #   for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

            # import sys
            # sys.exit()

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, S=7, B=hypers.NUM_BOXES, C=hypers.NUM_CLASSES, device=DEVICE)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f"epoch: {epoch} Train mAP: {mean_avg_prec}")
        k=0
        if (mean_avg_prec > 0.9) and (k==0):
            k=1
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=hypers.LOAD_MODEL_FILE, exit_training=True)
            # time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)
