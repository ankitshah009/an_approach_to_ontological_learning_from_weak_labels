import torch
import numpy as np
from sklearn import metrics


def evaluate_model_stats(data_loader, model, reduction='weighted'):
    
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
            _, out1_1, out1_2 = model.forward(input) # model output
            # _, _, out1_1, out1_2, out2_1, out2_2 = model.forward(input1[0:batch_size], input1[batch_size::]) # model output
            #targets = (target1_1[0:batch_size], target1_1[batch_size::], target1_2[0:batch_size], target1_2[batch_size::])
            
            sigmoid = torch.nn.Sigmoid()
            out1_1 = sigmoid(out1_1)
            # print(out1_1)

            # complete_outputs_1.append(torch.cat((out1_1, out2_1)))
            complete_outputs_1.append(out1_1)
            complete_targets_1.append(target1_1)
            
            # complete_outputs_2.append(torch.cat((out1_2, out2_2)))
            complete_outputs_2.append(out1_2)
            complete_targets_2.append(target1_2)
    
    
    complete_outputs_1 = torch.cat(complete_outputs_1, 0)
    complete_targets_1 = torch.cat(complete_targets_1, 0)
    #print(torch.sum(complete_targets_1, dim=0))
    
    complete_outputs_2 = torch.cat(complete_outputs_2, 0)
    complete_targets_2 = torch.cat(complete_targets_2, 0)
    
    num_classes_1 = complete_outputs_1.shape[-1]
    num_classes_2 = complete_outputs_2.shape[-1]
    
    # Move to CPU
    complete_targets_1 = complete_targets_1.detach().cpu().numpy()
    complete_outputs_1 = complete_outputs_1.detach().cpu().numpy()
    
    complete_targets_2 = complete_targets_2.detach().cpu().numpy()
    complete_outputs_2 = complete_outputs_2.detach().cpu().numpy()
    
    # Average outputs over entire audio clip
    output_1_avg = np.zeros((int(complete_outputs_1.shape[0]/10), complete_outputs_1.shape[1]))
    output_2_avg = np.zeros((int(complete_outputs_2.shape[0]/10), complete_outputs_2.shape[1]))
    for i in range(int(complete_outputs_1.shape[0]/10)):
        output_1_avg[i] = np.mean(complete_outputs_1[10*i:10*(i+1)], axis=0)
        output_2_avg[i] = np.mean(complete_outputs_2[10*i:10*(i+1)], axis=0)
    
    tot1 = np.sum(complete_targets_1[0::10])
    pos_weights_1 = np.sum(complete_targets_1[0::10], axis=0)
    neg_weights_1 = complete_targets_1[0::10].shape[0] - pos_weights_1
    tot2 = np.sum(complete_targets_2[0::10])
    pos_weights_2 = np.sum(complete_targets_2[0::10], axis=0)
    neg_weights_2 = complete_targets_2[0::10].shape[0] - pos_weights_2
    
    # print(neg_weights_1/pos_weights_1)
    # print(neg_weights_2/pos_weights_2)
    
    weights_1 = pos_weights_1 / tot1
    weights_2 = pos_weights_2 / tot2
    
    # Level 1 Average precision, AUC metrics
    average_precision_1 = np.zeros((num_classes_1, ))
    auc_1 = np.zeros((num_classes_1, ))
    for i in range(num_classes_1):
        average_precision_1[i] = metrics.average_precision_score(complete_targets_1[0::10, i], output_1_avg[:, i])
        auc_1[i] = metrics.roc_auc_score(complete_targets_1[0::10, i], output_1_avg[:, i], average = None)

    # Level 2 Average precision, AUC metrics
    average_precision_2 = np.zeros((num_classes_2, ))
    auc_2 = np.zeros((num_classes_2, ))
    for i in range(num_classes_2):
        average_precision_2[i] = metrics.average_precision_score(complete_targets_2[0::10, i], output_2_avg[:, i])
        auc_2[i] = metrics.roc_auc_score(complete_targets_2[0::10, i], output_2_avg[:, i], average = None)
        
    if(reduction=='average'):
        mAP_1 = np.mean(average_precision_1)
        mauc_1 = np.mean(auc_1)

        mAP_2 = np.mean(average_precision_2)
        mauc_2 = np.mean(auc_2)
        
    elif(reduction=='weighted'):
        mAP_1 = np.sum(weights_1*average_precision_1)
        mauc_1 = np.sum(weights_1*auc_1)
        
        mAP_2 = np.sum(weights_2*average_precision_2)
        mauc_2 = np.sum(weights_2*auc_2)
        
    elif(reduction=='none'):
        return average_precision_1, auc_1, complete_outputs_1, average_precision_2, auc_2, complete_outputs_2
    
    return mAP_1, mauc_1, complete_outputs_1, mAP_2, mauc_2, complete_outputs_2