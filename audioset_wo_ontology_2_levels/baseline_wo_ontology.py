import os
import sys
from pdb import set_trace as bp

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
from models_wo_ontology import SiameseNet_OntologicalLayer
from loss_wo_ontology import Ontological_Loss
from dataloaders_wo_ontology import AudioSet_Siamese, AudioSet_Siamese_Eval
from sklearn import metrics

# ########################### parameters #################################
# Define Loss function
lambda1 = 1
lambda2 = 1
batch_size = 64

lambda3 = 1
# criterion = Ontological_Loss(lambda1, lambda2, lambda3)

# Define Optimizer
learningRate = 2e-3
# weightDecay = 1e-3
# weightDecay = 5e-5
weightDecay = 1e-4
model_num = 'woOntM_levelFlatten' + '_wD_' + str(weightDecay) + '_b_' + str(batch_size) + '/'
# model_num = 'temp/'
print(model_num)
# ########################################################################

# GPU
cuda = torch.cuda.is_available()
print("cuda: ", cuda)
device = torch.device("cuda" if cuda else "cpu")


# Load data
data_dir = '../data/'

sounds_data = np.load(data_dir + 'audioset_train_data.npy', allow_pickle=True)
class1_index = np.load(data_dir + 'audioset_train_labels_1.npy', allow_pickle=True)
class2_index = np.load(data_dir + 'audioset_train_labels_2.npy', allow_pickle=True)

# clip_lengths = [clip.shape[0] for clip in sounds_data]
# print(clip_lengths)

# Dataloader
train_data = AudioSet_Siamese(sounds_data, class1_index, class2_index, 42, 7)
train_args = dict(shuffle = True, batch_size = batch_size, num_workers=8, pin_memory=True)
train_loader = DataLoader(train_data, **train_args)

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

num_label = train_data.logits_1.shape[0] + train_data.logits_2.shape[0]
pos_weights_1 = np.sum(train_data.logits_1, axis=0)
# neg_weights_1 = train_data.logits_1.shape[0] - pos_weights_1
neg_weights_1 = num_label - pos_weights_1
pos_weights_2 = np.sum(train_data.logits_2, axis=0)
# neg_weights_2 = train_data.logits_2.shape[0] - pos_weights_2
neg_weights_2 = num_label - pos_weights_2
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

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.66, min_lr=1e-6, verbose=True, patience=3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

### Train ###
def train_model(train_loader, model, epoch, writer):
    
    training_loss = 0
    training_acc_1 = 0
    training_acc_2 = 0
    
    # Set model in 'Training mode'
    model.train()
    
    # enumerate mini batches
    with tqdm(train_loader, ) as t_epoch:
        for i, (input, target1, target2) in enumerate(t_epoch):
            t_epoch.set_description(f"Epoch{epoch}")
            
            # Move to GPU
            input = input.to(device).float()
            target1 = target1.to(device)
            target2 = target2.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            
            # Model output
            outputs = model.forward(input) 
            targets = torch.cat((target1, target2), 1)
            
            # Loss/Backprop
            loss = criterion(outputs, targets)                           
            loss.backward()
            optimizer.step()     

            training_loss += loss.item()
            t_epoch.set_postfix(loss=loss.item())
            
            torch.cuda.empty_cache()
            del input
            del target1, target2
            del loss
    
    training_loss /= len(train_loader)
    writer.add_scalar("Loss/train", training_loss, epoch)  
    
    return training_loss


# Validation
def evaluate_model(val_loader, model, epoch, writer):
        
    val_loss = 0

    # Set model in validation mode
    model.eval()
    
    for i, (input, target1, target2) in enumerate(val_loader):
        with torch.no_grad():
            
            # Move to GPU
            input = input.to(device).float()
            target1 = target1.to(device)
            target2 = target2.to(device)
            
            # Model Output
            outputs = model.forward(input) # model output
            targets = torch.cat((target1, target2), 1)
            
            # Val loss
            loss = criterion(outputs, targets)            
            val_loss += loss.item()
    
    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)  

    return loss.item()

def evaluate_model_stats(data_loader, model, reduction='weighted'):
    
    model.eval()
    # device = torch.device("cuda:0")
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    complete_outputs = []
    complete_targets = []
    complete_targets_1 = []
    
    complete_outputs_2 = []
    complete_targets_2 = []
    
    # Evaluate test set in batches
    for i, (input, target1, target2) in enumerate(data_loader):
        with torch.no_grad():
            
            # Move to GPU
            input = input.to(device).float()
            target1 = target1.to(device)
            target2 = target2.to(device)
            
            # Model Output
            _, out = model.forward(input) # model output
            target = torch.cat((target1, target2), 1)
            
            complete_outputs.append(out)
            complete_targets.append(target)
    
    # Concat batch results and move to cpu
    complete_outputs = torch.cat(complete_outputs, 0).detach().cpu().numpy()
    complete_targets = torch.cat(complete_targets, 0).detach().cpu().numpy()
    
    num_classes_1 = 42
    num_classes_2 = 7
    seg_per_clip = 10
    # seg_per_clip = 1
    # Average outputs over entire audio clip
    output_avg = np.zeros((int(complete_outputs.shape[0]/seg_per_clip), complete_outputs.shape[1]))
    for i in range(int(complete_outputs.shape[0]/seg_per_clip)):
        output_avg[i] = np.mean(complete_outputs[seg_per_clip*i:seg_per_clip*(i+1)], axis=0)
    
    # separate targets and outputs into 2 levels
    complete_targets_1, complete_targets_2 = complete_targets[:, :num_classes_1], complete_targets[:, num_classes_1:]
    output_avg_1, output_avg_2 = output_avg[:, :num_classes_1], output_avg[:, num_classes_1:]
    
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
        if np.sum(complete_targets_1[:, i] != 0):
            average_precision_1[i] = metrics.average_precision_score(complete_targets_1[0::seg_per_clip, i], output_avg_1[:, i])
            auc_1[i] = metrics.roc_auc_score(complete_targets_1[0::seg_per_clip, i], output_avg_1[:, i], average = None)
            

    # Level 2 Average precision, AUC metrics
    average_precision_2 = np.zeros((num_classes_2, ))
    auc_2 = np.zeros((num_classes_2, ))
    for i in range(num_classes_2):
        average_precision_2[i] = metrics.average_precision_score(complete_targets_2[0::seg_per_clip, i], output_avg_2[:, i])
        auc_2[i] = metrics.roc_auc_score(complete_targets_2[0::seg_per_clip, i], output_avg_2[:, i], average = None)
        
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
        return average_precision_1, auc_1, complete_outputs, average_precision_2, auc_2, complete_outputs_2
    
    return mAP_1, mauc_1, complete_outputs, mAP_2, mauc_2, complete_outputs_2

val_mAP_1, val_auc_1,  _, val_mAP_2, val_auc_2,  _ = evaluate_model_stats(eval_loader, model)


print(val_mAP_1)
print(val_auc_1)
print(val_mAP_2)
print(val_auc_2)

# training 1 -----------------------------------------------------------------------

# Save dirs
base_dir = '/scratch/yyzdata/dl/project/'
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
epochs = 80
sum_map_auc = []
for epoch in range(epochs):
    
    # Train + Validation
    training_loss = train_model(train_loader, model, epoch, writer)
    val_loss = evaluate_model(val_loader, model, epoch, writer)
    
    # Val Stats
    val_mAP_1, val_auc_1,  _, val_mAP_2, val_auc_2,  _ = evaluate_model_stats(eval_loader, model)
    sum_map_auc.append(val_mAP_1 + val_auc_1 + val_mAP_2 + val_auc_2)
    
    # scheduler.step(val_mAP_1 + val_mAP_2)
    scheduler.step()
    
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

# Test Data -----------------------------------------------------------------------
test_ep = np.argmax(sum_map_auc)
# test_ep = 73
print("test_ep", test_ep)

sounds_data = np.load(data_dir + 'audioset_test_data.npy', allow_pickle=True)
class1_index = np.load(data_dir + 'audioset_test_labels_1.npy', allow_pickle=True)
class2_index = np.load(data_dir + 'audioset_test_labels_2.npy', allow_pickle=True)

# sounds_data = np.load(data_dir + 'audioset_strong_eval1_data.npy', allow_pickle=True)
# class1_index = np.load(data_dir + 'audioset_strong_eval1_labels_1.npy', allow_pickle=True)
# class2_index = np.load(data_dir + 'audioset_strong_eval1_labels_2.npy', allow_pickle=True)

eval_data = AudioSet_Siamese_Eval(sounds_data, class1_index, class2_index, 42, 7)
eval_args = dict(shuffle = False, batch_size = batch_size, num_workers=8, pin_memory=True)
eval_loader = DataLoader(eval_data, **eval_args)

# Load model
model_save_dir = '/scratch/yyzdata/dl/project/models/woOntM_levelFlatten_wD_0.0001_b_64/'
# model_save_dir = model_dir

model_num = 'epoch0.pt'
model = torch.load(model_save_dir + model_num) 
model.to(device)

# Load saved weights
weights_dir = 'epoch' + str(test_ep) + '.pt'
state = torch.load(model_save_dir + weights_dir)
model.load_state_dict(state['model_state_dict'])
model.eval()

val_mAP_1, val_auc_1,  _, val_mAP_2, val_auc_2,  _ = evaluate_model_stats(eval_loader, model)
print("Validation mAP_1/AUC_1: "+str(val_mAP_1)+"/"+str(val_auc_1)+
             ", Validation mAP_2/AUC_2: "+str(val_mAP_2)+"/"+str(val_auc_2))

test_mAP_1, test_AUC_1, outputs_1, test_mAP_2, test_AUC_2, outputs_2 = evaluate_model_stats(eval_loader, model, reduction='none')

print(test_mAP_1)
print(test_AUC_1)
print(test_mAP_2)
print(test_AUC_2)