from pdb import set_trace as bp
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

class AudioSet_Siamese(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2  
        self.context = 1
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        num_clips = len(data)
        
        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i]] = 1 
            self.logits_2[i][self.labels2[i]] = 1
        
        # fire 9, glass 11, human group action 15, silence
        cls_id = 0
        print("num of instances", self.logits_1[:, cls_id].sum())
        clip_contain_fire = np.where(self.logits_1[:, cls_id] == 1)[0]
        num_label = []
        for clip_id in clip_contain_fire:
            num_label.append(self.logits_1[clip_id, :].sum())
        print(num_label)
        bp()
        
        # print("self.logits_1", self.logits_1[:, 42].sum())
    def __len__(self):
        
        return int(self.length)
        
    def __getitem__(self, idx):
        
        x1 = self.data[idx]
        class_idx = int(idx / 10)
        
        # Superclass
        x1_superclass = self.logits_2[class_idx]
        # Subclass
        x1_subclass = self.logits_1[class_idx]
                           
        return x1.flatten(), x1_subclass, x1_superclass        

    
class AudioSet_Siamese_Eval(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2  
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        num_clips = len(data)

        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i]] = 1 
            self.logits_2[i][self.labels2[i]] = 1
        
        # print("self.logits_1 eval", self.logits_1[:, 42].sum())
    def __len__(self):
        
        return self.length
        
    def __getitem__(self, index):
        
        x1 = self.data[index]
        
        return x1.flatten(), self.logits_1[int(index/10)], self.logits_2[int(index/10)]
        
        
        