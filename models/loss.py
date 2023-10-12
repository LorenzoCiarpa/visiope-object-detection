import torch
import torch.nn as nn 
from dataprocess.dataprocess import *

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = 5, .5

    def forward(self, predictions, target ):
        predictions = predictions.reshape(-1, self.S, self.S, self.B*5+self.C)
        ious = torch.cat(
            (intersection_over_union(predictions[...,self.C+1:self.C+5], target[...,self.C+1:self.C+5]).unsqueeze(0),  #unsqueeze(0) works only if b_size == 1
            intersection_over_union(predictions[...,self.C+6:self.C+10], target[...,self.C+1:self.C+5]).unsqueeze(0)) # target = [1, 1, x, y, w, h, 0,0,0,0,0]
                                                                                                                      #pred = [0.8, 0, 0.6, 0.8, 0.5, 0.7, 1, 0.5, 0.4,
        )

        '''
            CASE 1 CLASS:
            C+1:C+5 -> 2:6 (4) ELEMENTS
            C+6:C+10 -> 7:11 (4) ELEMENTS
            element in position [C+5] is excluded
        '''

        _, bestbox = torch.max(ious, dim=0)
        exists_box = target[...,self.C:self.C+1]

        box_predictions =  exists_box * ( (1-bestbox)*predictions[...,self.C+1:self.C+5] + bestbox*predictions[...,self.C+6:self.C+10] )
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6)) # WOW
        box_targets = exists_box * target[...,self.C+1:self.C+5] #Just these 4 because it can predicts at most 1 box for each cell
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) # rad(w), rad(h)

        # print("box_predictions: ", box_targets.shape)
        # print("box_targets: ", box_targets.shape)
        # print("box_predictions_flatten: ", torch.flatten(box_predictions, end_dim=-2).shape)
        # print("box_targets_flatten: ", torch.flatten(box_targets, end_dim=-2).shape)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2), # batch_size*S*S, 4 e.g. (16*7*7=784, 4)
            torch.flatten(box_targets, end_dim=-2)      # batch_size*S*S, 4
        )
        '''
        object_loss is greater if my model predicted the correct box between the B possible boxes
        C:C+1 and C+5:C+6 are 1 if the chosen box is respectively the first or the second
        '''
        object_loss = self.mse(
            torch.flatten(exists_box * ( (1-bestbox)*predictions[...,self.C:self.C+1] + bestbox*predictions[...,self.C+5:self.C+6] )),
            torch.flatten(exists_box)
        )
        '''
        no_object_loss exists when no object are in the image
        else it penalize the loss when the model predicted a box or worse if the model predicted both
        '''
        no_object_loss=self.mse(
            torch.flatten((1-exists_box) * predictions[...,self.C:self.C+1], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,self.C:self.C+1], end_dim=-2)
        ) + self.mse(
            torch.flatten((1-exists_box) * predictions[...,self.C+5:self.C+6], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,self.C:self.C+1], end_dim=-2)
        )
        '''
        penilize if the model predicted the wrong class
        NOTE: this can be remove if the class is just one
        '''
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[...,:self.C], end_dim=-2),
            torch.flatten(exists_box * target[...,:self.C], end_dim=-2)
        )
        loss = (self.lambda_coord*box_loss + object_loss + self.lambda_noobj*no_object_loss + class_loss)
        return loss
      
      
class YoloLossMultiBoxes(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLossMultiBoxes, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = 5, .5

    def forward(self, predictions, target ):
        predictions = predictions.reshape(-1, self.S, self.S, self.B*5+self.C)
        intersection_tuples = ()

        for i_1 in range(self.B):
            i_o_u = intersection_over_union(predictions[...,self.C+(1+i_1*5):self.C+(5+i_1*5)], target[...,self.C+1:self.C+5]).unsqueeze(0)
            intersection_tuples += (i_o_u,)

        # i_o_u = intersection_over_union(target[...,self.C+1:self.C+5], target[...,self.C+1:self.C+5]).unsqueeze(0)
        # intersection_tuples += (i_o_u,)

        ious = torch.cat(intersection_tuples) # B X batch_size X S X S X self.B*5+self.C

        _, bestbox = torch.max(ious, dim=0) # Why on dimension 0 and not -1?
        exists_box = target[...,self.C:self.C+1]

        # extraction_classes = target[...,0:self.C]
        '''
        le prime C elems sono relativi a le probabilità per ogni classe,

        '''



        # print(f'bestbox: {bestbox}')
        # print(f'exists_box shape: {exists_box.shape}, bestbox shape: {bestbox.shape}, target shape: {target.shape}, target POST shape: {target[...,0].shape}, predictions shape: {predictions.shape}')

        # predictions = [
        #   [
        #     [[1, 0, 0.5, 0.7, 0.9, 0.3, 0, 0.3, 0.4, 0.6, 0.4, 0, 0.3, 0.4, 0.6, 0.4]], [[1, 0, 0.5, 0.7, 0.9, 0.3, 0, 0.3, 0.4, 0.6, 0.4, 0, 0.3, 0.4, 0.6, 0.4]]
        #   ],
        #   [
        #     [[1, 0, 0.5, 0.7, 0.9, 0.3, 0, 0.3, 0.4, 0.6, 0.4, 0, 0.3, 0.4, 0.6, 0.4]], [[1, 0, 0.5, 0.7, 0.9, 0.3, 0, 0.3, 0.4, 0.6, 0.4, 0, 0.3, 0.4, 0.6, 0.4]]
        #   ]
        # ]

        # bestbox = [
        #   [
        #     [0], [1]
        #   ],
        #   [
        #     [2], [1]
        #   ]
        # ]

        # exists_box = [
        #   [
        #     [0], [1]
        #   ],
        #   [
        #     [0], [0]
        #   ]
        # ]

        # preds_box = (1 - bestbox)
        # preds_box[preds_box != 1] = 0
        # box_predictions = preds_box * predictions[...,self.C+(1):self.C+(5)]

        pred_shape = predictions.shape[0:3]
        pred_shape = list(pred_shape)
        pred_shape.append(4)

        box_predictions = torch.zeros(pred_shape).to(hypers.DEVICE) # batch_size x S x S x 4

        for i_1 in range(self.B): #prendo solo quello relativo all'argmax
            # preds_box = ((1 + i_1) - bestbox)
            preds_box = bestbox.detach().clone()
            if i_1 == 0:
              preds_box[preds_box == i_1] = -1
              preds_box[preds_box != -1] = 0
              preds_box[preds_box == -1] = 1
            else:
              preds_box[preds_box != i_1] = 0
              preds_box[preds_box == i_1] = 1

            box_predictions += preds_box * predictions[...,self.C+(1+i_1*5):self.C+(5+i_1*5)]

            # box_predictions += i_1 * predictions[...,self.C+(1+i_1*5):self.C+(5+i_1*5)] # da eliminare

        box_predictions = exists_box * box_predictions
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6)) # WOW
        box_targets = exists_box * target[...,self.C+1:self.C+5] #Just these 4 because it can predicts at most 1 box for each cell
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) # rad(w), rad(h)


        '''
            CASE 1 CLASS:
            C+1:C+5 -> 2:6 (4) ELEMENTS
            C+6:C+10 -> 7:11 (4) ELEMENTS
            element in position [C+5] is excluded
        '''

#         box_predictions =  exists_box * ( (1-bestbox)*predictions[...,self.C+1:self.C+5] + bestbox*predictions[...,self.C+6:self.C+10] )
#         box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6)) # WOW
#         box_targets = exists_box * target[...,self.C+1:self.C+5] #Just these 4 because it can predicts at most 1 box for each cell
#         box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) # rad(w), rad(h)

        # print("box_predictions: ", box_targets.shape)
        # print("box_targets: ", box_targets.shape)
        # print("box_predictions_flatten: ", torch.flatten(box_predictions, end_dim=-2).shape)
        # print("box_targets_flatten: ", torch.flatten(box_targets, end_dim=-2).shape)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2), # batch_size*S*S, 4 e.g. (16*7*7=784, 4)
            torch.flatten(box_targets, end_dim=-2)      # batch_size*S*S, 4
        )
        '''
        object_loss is greater if my model predicted the correct box between the B possible boxes
        C:C+1 and C+5:C+6 are 1 if the chosen box is respectively the first or the second
        '''

        # print(f'exists_box shape: {exists_box.shape}, bestbox shape: {bestbox.shape}, target shape: {target.shape}, target POST shape: {target[...,0:1].shape}, predictions shape: {predictions.shape} predictions POST shape: {predictions[...,self.C:self.C+1].shape}')
        pred_shape = predictions.shape[0:3]
        pred_shape = list(pred_shape)
        pred_shape.append(1)
        preds = torch.zeros(pred_shape).to(hypers.DEVICE) # batch_size x S x S x 1

        for i_1 in range(self.B): #prendo solo quello relativo all'argmax
            # preds_box = ((1 + i_1) - bestbox)
            preds_box = bestbox.detach().clone()
            if i_1 == 0:
              preds_box[preds_box == i_1] = -1
              preds_box[preds_box != -1] = 0
              preds_box[preds_box == -1] = 1
            else:
              preds_box[preds_box != i_1] = 0
              preds_box[preds_box == i_1] = 1

            preds += preds_box * predictions[...,self.C+(i_1*5):self.C+(1+i_1*5)]

        preds = exists_box * preds


        # preds = target[...,0:1] * predictions[...,self.C:self.C+1]
        # for i_1 in range(1, self.B): #prendo solo quello relativo all'argmax
        #     preds += target[...,i_1:i_1+1] * predictions[...,self.C+(i_1*5):self.C+(1+i_1*5)]
        # preds = exists_box * preds

        object_loss = self.mse(
            torch.flatten(preds),
            torch.flatten(exists_box)
        )
        '''
        no_object_loss exists when no object are in the image
        else it penalize the loss when the model predicted a box or worse if the model predicted both
        '''
        no_object_loss = self.mse(
                torch.flatten((1-exists_box) * predictions[...,self.C:self.C+1], end_dim=-2),
                torch.flatten((1-exists_box) * target[...,self.C:self.C+1], end_dim=-2)
            )
        for i_1 in range(1, self.B): #prendo solo quello relativo all'argmax
            no_object_loss += self.mse(
                torch.flatten((1-exists_box) * predictions[...,self.C+(i_1*5):self.C+(1+i_1*5)], end_dim=-2),
                torch.flatten((1-exists_box) * target[...,self.C:self.C+1], end_dim=-2)
            )
        '''
        penilize if the model predicted the wrong class
        NOTE: this can be remove if the class is just one
        '''
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[...,:self.C], end_dim=-2),
            torch.flatten(exists_box * target[...,:self.C], end_dim=-2)
        )
        loss = (self.lambda_coord*box_loss + object_loss + self.lambda_noobj*no_object_loss + class_loss)
        return loss