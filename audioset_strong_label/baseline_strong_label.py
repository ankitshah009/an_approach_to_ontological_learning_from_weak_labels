import os
import sys

# Model/Training related libraries
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Dataloader libraries
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision import transforms

# Custom models/loss/dataloaders/utilities
from models_strong_label import SiameseNet_OntologicalLayer
from loss_strong_label import Ontological_Loss
from dataloaders_strong_label import AudioSet_Siamese, AudioSet_Siamese_Eval
from sklearn import metrics
import scipy.io
import pdb


# GPU
cuda = torch.cuda.is_available()
print("cuda: ", cuda)
device = torch.device("cuda" if cuda else "cpu")

# Load data
data_dir = '../data/'
base_dir = '/scratch/yyzdata/dl/project/'
lambda1 = 1.5
lambda2 = 1
lambda3 = 0.25
batch_size = 128
epochs = 70
learningRate = 1e-3
weightDecay = 1e-1
model_num = 'complete_audio*20_weightedLoss' + '_b_' + str(batch_size) + '_l1_' + str(lambda1) + '_l2_' + str(lambda2) + '_l3_' + str(lambda3) + '/'
# model_num = 'temp/'

print("model_num", model_num)

# base_dir = './'
sounds_data = np.load(data_dir + 'audioset_train_data.npy', allow_pickle=True)
class1_index = np.load(data_dir + 'audioset_train_labels_1.npy', allow_pickle=True)
class2_index = np.load(data_dir + 'audioset_train_labels_2.npy', allow_pickle=True)

# Dataloader
train_data = AudioSet_Siamese(sounds_data, class1_index, class2_index, 42, 7)
train_args = dict(shuffle = True, batch_size = batch_size, num_workers=8, pin_memory=True)
train_loader = DataLoader(train_data, **train_args)

train_data.__getitem__(0)

sounds_data = np.load(data_dir + 'audioset_val_data.npy', allow_pickle=True)
class1_index = np.load(data_dir + 'audioset_val_labels_1.npy', allow_pickle=True)
class2_index = np.load(data_dir + 'audioset_val_labels_2.npy', allow_pickle=True)

val_data = AudioSet_Siamese(sounds_data, class1_index, class2_index, 42, 7)
val_args = dict(shuffle = False, batch_size = batch_size, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_data, **val_args)

eval_data = AudioSet_Siamese_Eval(sounds_data, class1_index, class2_index, 42, 7)
eval_args = dict(shuffle = False, batch_size = batch_size, num_workers=8, pin_memory=True, sampler=SequentialSampler(eval_data))
eval_loader = DataLoader(eval_data, **eval_args)

# Ontology Layer
M = np.asarray([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

M = M / np.sum(M, axis=1).reshape(-1, 1)

# Siamese Net Model
in_feature_dim = train_data.__getitem__(0)[0].shape[0]
model = SiameseNet_OntologicalLayer(7, 42, in_feature_dim, M)
model.to(device)

# Define Loss function

# Weights for BCE Loss 
tot1 = np.sum(train_data.logits_1)
pos_weights_1 = np.sum(train_data.logits_1, axis=0)
neg_weights_1 = train_data.logits_1.shape[0] - pos_weights_1
tot2 = np.sum(train_data.logits_2)
pos_weights_2 = np.sum(train_data.logits_2, axis=0)
neg_weights_2 = train_data.logits_2.shape[0] - pos_weights_2
label_weights_1 = torch.tensor(neg_weights_1/pos_weights_1).to(device)
label_weights_2 = torch.tensor(neg_weights_2/pos_weights_2).to(device)
# label_weights_1 = torch.tensor([ 14.64592934, 20.1536864,   48.32445521, 116.07471264,  50.05513784,
#   32.28594771,  10.41232493,  31.9095315,   36.03818182, 164.61788618,
#   43.57549234, 117.43604651, 123.97546012, 174.61206897,  53.76075269,
#  112.80446927,   2.06884604,  27.81329562,  24.98341837,  29.7254902,
#   40.32048682,   9.21614845,  48.56447689,  39.66067864, 123.21341463,
#    5.3699187,   28.65211063,  17.07542147,  74.1697417,   34.305026,
#  344.27118644, 112.17222222, 338.51666667,  63.05974843, 312.4,
#   52.4671916,    9.26246851,  27.9772404,  338.51666667,  15.90539419,
#   82.83127572,  72.01433692]).to(device)
# label_weights_2 = torch.tensor([ 7.48084929,  8.92255236,  1.72996516,  1.61569081, 19.59757331,  1.62648272, 7.55564889]).to(device)
criterion = Ontological_Loss(lambda1, lambda2, lambda3, label_weights_1, label_weights_2)

# Define Optimizer


optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.65, patience=2)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=1e-6, verbose=True, patience=2)

### Train ###
def train_model(train_loader, model, epoch, writer):
    
    training_loss = 0
    training_acc_1 = 0
    training_acc_2 = 0
    
    # Set model in 'Training mode'
    model.train()
    
    # enumerate mini batches
    with tqdm(train_loader, ) as t_epoch:
        for i, (input1, input2, target1_1, target1_2, target2_1, target2_2, pair_type) in enumerate(t_epoch):
            t_epoch.set_description(f"Epoch {epoch}")
            
            # Move to GPU
            input1 = input1.to(device).float()
            target1_1 = target1_1.to(device)
            target1_2 = target1_2.to(device)
            
            # print(target1_1.size())
            
            input2 = input2.to(device).float()
            target2_1 = target2_1.to(device)
            target2_2 = target2_2.to(device)
            
            # print(pair_type)
            pair_type = pair_type.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            
            # Model output
            outputs = model.forward(input1.float(), input2.float()) 
            targets = (target1_1, target1_2, target2_1, target2_2)
            
            # Loss/Backprop
            loss = criterion(outputs, targets, pair_type)                           
            loss.backward()
            optimizer.step()     

            training_loss += loss.item()
            t_epoch.set_postfix(loss=loss.item())
            
            torch.cuda.empty_cache()
            del input1, input2
            del target1_1, target1_2, target2_1, target2_2, pair_type
            del loss
    
    training_loss /= len(train_loader)
    writer.add_scalar("Loss/train", training_loss, epoch)  
    
    return training_loss


# Validation
def evaluate_model(val_loader, model, epoch, writer):
        
    val_loss = 0

    # Set model in validation mode
    model.eval()
    
    for i, (input1, input2, target1_1, target1_2, target2_1, target2_2, pair_type) in enumerate(val_loader):
        with torch.no_grad():
            
            # Move to GPU
            input1 = input1.to(device).float()
            target1_1 = target1_1.to(device)
            target1_2 = target1_2.to(device)
            
            input2 = input2.to(device).float()
            target2_1 = target2_1.to(device)
            target2_2 = target2_2.to(device)
            
            pair_type = pair_type.to(device)
            
            # Model Output
            outputs = model.forward(input1, input2) # model output
            targets = (target1_1, target1_2, target2_1, target2_2)
            
            # Val loss
            loss = criterion(outputs, targets, pair_type)            
            val_loss += loss.item()
    
    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)  

    return loss.item()

def evaluate_model_stats(data_loader, model, batch_size, reduction='weighted'):
    
    model.eval()
    cuda = torch.cuda.is_available()
    print("cuda: ", cuda)
    device = torch.device("cuda" if cuda else "cpu")
    
    complete_outputs_1 = []
    complete_targets_1 = []
    
    complete_outputs_2 = []
    complete_targets_2 = []
    
    # Evaluate test set in batches
    for i, (input1, target1_1, target1_2) in enumerate(data_loader):
        with torch.no_grad():
            
            # Move to GPU
            input1 = input1.to(device).float()
            target1_1 = target1_1.to(device)
            target1_2 = target1_2.to(device)
            
            batch_size = int(input1.shape[0]/2)
            
            # Model Output
            
            # import pdb; pdb.set_trace()
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
    
    
    # Concat batch results 
    complete_outputs_1 = torch.cat(complete_outputs_1, 0)
    complete_targets_1 = torch.cat(complete_targets_1, 0)
    
    complete_outputs_2 = torch.cat(complete_outputs_2, 0)
    complete_targets_2 = torch.cat(complete_targets_2, 0)
    
    # print(complete_outputs_1)
    # print(complete_targets_1)
    
    num_classes_1 = complete_outputs_1.shape[-1]
    num_classes_2 = complete_outputs_2.shape[-1]
    
    # Move to CPU
    complete_targets_1 = complete_targets_1.detach().cpu().numpy()
    complete_outputs_1 = complete_outputs_1.detach().cpu().numpy()
    
    complete_targets_2 = complete_targets_2.detach().cpu().numpy()
    complete_outputs_2 = complete_outputs_2.detach().cpu().numpy()
    
    # seg_per_clip = 10
    seg_per_clip = 1
    # Average outputs over entire audio clip
    output_1_avg = np.zeros((int(complete_outputs_1.shape[0]/seg_per_clip), complete_outputs_1.shape[1]))
    output_2_avg = np.zeros((int(complete_outputs_2.shape[0]/seg_per_clip), complete_outputs_2.shape[1]))
    for i in range(int(complete_outputs_1.shape[0]/seg_per_clip)):
        output_1_avg[i] = np.mean(complete_outputs_1[seg_per_clip*i:seg_per_clip*(i+1)], axis=0)
        output_2_avg[i] = np.mean(complete_outputs_2[seg_per_clip*i:seg_per_clip*(i+1)], axis=0)
    
    tot1 = np.sum(complete_targets_1[0::seg_per_clip])
    pos_weights_1 = np.sum(complete_targets_1[0::seg_per_clip], axis=0)
    neg_weights_1 = complete_targets_1[0::seg_per_clip].shape[0] - pos_weights_1
    tot2 = np.sum(complete_targets_2[0::seg_per_clip])
    pos_weights_2 = np.sum(complete_targets_2[0::seg_per_clip], axis=0)
    neg_weights_2 = complete_targets_2[0::seg_per_clip].shape[0] - pos_weights_2
    
    # print(neg_weights_1/pos_weights_1)
    # print(neg_weights_2/pos_weights_2)
    
    weights_1 = pos_weights_1 / tot1
    weights_2 = pos_weights_2 / tot2
            
    # Level 1 Average precision, AUC metrics
    average_precision_1 = np.zeros((num_classes_1, ))
    auc_1 = np.zeros((num_classes_1, ))
    for i in range(num_classes_1):
        average_precision_1[i] = metrics.average_precision_score(complete_targets_1[0::seg_per_clip, i], output_1_avg[:, i])
        auc_1[i] = metrics.roc_auc_score(complete_targets_1[0::seg_per_clip, i], output_1_avg[:, i], average = None)

    # Level 2 Average precision, AUC metrics
    average_precision_2 = np.zeros((num_classes_2, ))
    auc_2 = np.zeros((num_classes_2, ))
    for i in range(num_classes_2):
        average_precision_2[i] = metrics.average_precision_score(complete_targets_2[0::seg_per_clip, i], output_2_avg[:, i])
        auc_2[i] = metrics.roc_auc_score(complete_targets_2[0::seg_per_clip, i], output_2_avg[:, i], average = None)
        
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

print(train_data.seg_per_clip)
print(val_data.data.shape)
# sounds_data = np.load(data_dir + 'audioset_yamnet_train_data.npy', allow_pickle=True)
# print(sounds_data.shape)
print(label_weights_1)

# Save dirs
model_dir = base_dir + 'models/' + model_num
runs_dir = base_dir + 'runs/' + model_num

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

# Tensorboard logging
writer = SummaryWriter(runs_dir)
torch.backends.cudnn.benchmark = True

# Training loop
for epoch in range(epochs):
    
    # Train + Validation
    training_loss = train_model(train_loader, model, epoch, writer)
    val_loss = evaluate_model(val_loader, model, epoch, writer)
    
    # Val Stats
    val_mAP_1, val_auc_1, _, val_mAP_2, val_auc_2, _ = evaluate_model_stats(eval_loader, model, batch_size)

    scheduler.step(val_mAP_1 + val_auc_1 + val_mAP_2 + val_auc_2)
    
    # Print log of accuracy and loss
    print("Epoch: "+str(epoch)+", Training loss: "+str(training_loss)+", Validation loss: "+str(val_loss)+", Validation mAP_1/AUC_1: "+str(val_mAP_1)+"/"+str(val_auc_1)+
             ", Validation mAP_2/AUC_2: "+str(val_mAP_2)+"/"+str(val_auc_2))
    
    writer.add_scalar("mAP_1/val", val_mAP_1, epoch)
    writer.add_scalar("AUC_1/val", val_auc_1, epoch)
    writer.add_scalar("mAP_2/val", val_mAP_2, epoch)
    writer.add_scalar("AUC_2/val", val_auc_2, epoch)
    
    # Save model checkpoint
    model_filename = model_dir + 'epoch' + str(epoch) + '.pt'
    if(epoch == 0):
        torch.save(model, model_filename)
    else:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': training_loss,}, model_filename)