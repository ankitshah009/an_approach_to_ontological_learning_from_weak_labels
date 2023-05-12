import torch
import numpy as np
from sklearn import metrics


def evaluate_model_stats(data_loader, model):
    
    model.eval()
    device = torch.device("cuda:0")
    
    complete_outputs_1 = []
    complete_targets_1 = []
    
    complete_outputs_2 = []
    complete_targets_2 = []
    
    for i, (input1, target1_1, target1_2) in enumerate(data_loader):
        with torch.no_grad():
            
            # Move to GPU
            input1 = input1.to(device).float()
            target1_1 = target1_1.to(device)
            target1_2 = target1_2.to(device)
            
            batch_size = int(input1.shape[0]/2)
            
            # Model Output
            _, _, out1_1, out1_2, out2_1, out2_2 = model.forward(input1[0:batch_size], input1[batch_size::]) # model output
            #targets = (target1_1[0:batch_size], target1_1[batch_size::], target1_2[0:batch_size], target1_2[batch_size::])
            
            sigmoid = torch.nn.Sigmoid()
            out1_1 = sigmoid(out1_1)
            out2_1 = sigmoid(out2_1)
            # print(out1_1)

            complete_outputs_1.append(torch.cat((out1_1, out2_1)))
            complete_targets_1.append(target1_1)
            
            complete_outputs_2.append(torch.cat((out1_2, out2_2)))
            complete_targets_2.append(target1_2)
    
    
    complete_outputs_1 = torch.cat(complete_outputs_1, 0)
    complete_targets_1 = torch.cat(complete_targets_1, 0)
    #print(torch.sum(complete_targets_1, dim=0))
    
    complete_outputs_2 = torch.cat(complete_outputs_2, 0)
    complete_targets_2 = torch.cat(complete_targets_2, 0)
    
    mAP_1, auc_1, mAP_2, auc_2 = compute_stats(complete_outputs_1, complete_targets_1, complete_outputs_2, complete_targets_2)
    return mAP_1, auc_1, mAP_2, auc_2
    


def compute_stats(output_1, target_1, output_2, target_2):
    
    # _, _, out1_1, out1_2, out2_1, out2_2 = outputs
    # target1_1, target1_2, target2_1, target2_2 = targets
    
    num_classes_1 = output_1.shape[-1]
    num_classes_2 = output_2.shape[-1]
    
#     target_1 = torch.cat((target1_1, target2_1))
#     output_1 = torch.cat((out1_1, out2_1))
    
#     target_2 = torch.cat((target1_2, target2_2))
#     output_2 = torch.cat((out1_2, out2_2))
    
    # Move to CPU
    target_1 = target_1.detach().cpu().numpy()
    output_1 = output_1.detach().cpu().numpy()
    
    target_2 = target_2.detach().cpu().numpy()
    output_2 = output_2.detach().cpu().numpy()
    
    output_1 = np.mean(output_1.reshape(10, -1, num_classes_1), axis=1)
    output_2 = np.mean(output_2.reshape(10, -1, num_classes_2), axis=1)
    
    print(output_1.shape)
    
    # Level 1 Average precision, AUC metrics
    average_precision_1 = np.zeros((num_classes_1, ))
    auc_1 = np.zeros((num_classes_1, ))
    for i in range(num_classes_1):
        if i != 29 : # not present in val unbal set
            average_precision_1[i] = metrics.average_precision_score(target_1[:, i], output_1[:, i])
            auc_1[i] = metrics.roc_auc_score(target_1[:, i], output_1[:, i], average = None)

    # Level 2 Average precision, AUC metrics
    average_precision_2 = np.zeros((num_classes_2, ))
    auc_2 = np.zeros((num_classes_2, ))
    for i in range(num_classes_2):
        average_precision_2[i] = metrics.average_precision_score(target_2[:, i], output_2[:, i])
        auc_2[i] = metrics.roc_auc_score(target_2[:, i], output_2[:, i], average = None)
        
    mAP_1 = np.sum(average_precision_1)/(num_classes_1-1)
    auc_1 = np.sum(auc_1)/(num_classes_1-1)
    
    mAP_2 = np.mean(average_precision_2)
    auc_2 = np.mean(auc_2)
        
    return mAP_1, auc_1, mAP_2, auc_2
    
        
        