import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

class AudioSet_Siamese(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        # self.data = np.concatenate(data, axis=0).astype('float')
        # self.seg_per_clip = data[0].shape[0]
        # import pdb; pdb.set_trace()
        self.data = data.astype('float')
        self.seg_per_clip = 1
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2  
        self.context = 1
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        num_clips = len(data)
        
        self.same_subclass = []
        self.diff_subclass_same_superclass = []
        self.diff_superclass = []
            
        
        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i].astype('int')] = 1 
            self.logits_2[i][self.labels2[i].astype('int')] = 1
        
        # self.same_subclass = [np.where( np.all(self.logits_1[i] == self.logits_1, axis=1) )[0] for i in range(num_clips)]
        # self.diff_subclass_same_superclass = [np.where( np.logical_and( np.logical_not( np.all(self.logits_1[i] == self.logits_1, axis=1)),  
        #                                                                np.all(self.logits_2[i] == self.logits_2, axis=1)) )[0] for i in range(num_clips)]
        # self.diff_superclass = [ np.where( np.logical_not( np.all(self.logits_2[i] == self.logits_2, axis=1) ) )[0] for i in range(num_clips) ]
        
        
    def __len__(self):
        
        return int(self.length * 20)
        
    def __getitem__(self, idx):
        
        idx = int(idx / 20)
        class_idx = int(idx/self.seg_per_clip)
        idx2 = np.random.choice(np.arange(self.length))
        class_idx2 = int(idx2/self.seg_per_clip)
        
        x1 = self.data[idx]
        x2 = self.data[idx2]
        
        # Superclass
        x1_superclass = self.logits_2[class_idx]
        x2_superclass = self.logits_2[class_idx2]
        
        # Subclass
        x1_subclass = self.logits_1[class_idx]
        x2_subclass = self.logits_1[class_idx2]   
        
        # print(x1_superclass)
        # print(x2_superclass)
        # print(x1_subclass)
        # print(x2_subclass)
        pair_type = np.logical_or(x1_superclass, x2_superclass).sum() + np.logical_or(x1_subclass, x2_subclass).sum()
        # import pdb; pdb.set_trace()
        # pair_type = np.linalg.norm(x1_superclass - x2_superclass) + np.linalg.norm(x1_subclass - x2_subclass) * 3
                           
        return x1.flatten(), x2.flatten(), x1_subclass, x1_superclass, x2_subclass, x2_superclass, pair_type        

    
class AudioSet_Siamese_Eval(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        # self.data = np.concatenate(data, axis=0).astype('float')
        # self.seg_per_clip = data[0].shape[0]
        self.data = data.astype('float')
        self.seg_per_clip = 1
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2  
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        num_clips = len(data)

        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i].astype('int')] = 1 
            self.logits_2[i][self.labels2[i].astype('int')] = 1
        
    def __len__(self):
        
        return self.length
        
    def __getitem__(self, index):
        
        # if (index % 10 == 0):
        #     x1 = np.repeat(self.data[index:index+2], [2, 1], axis=0) 
        # elif (index % 10 == 9):
        #     x1 = np.repeat(self.data[index-1:index+1], [1, 2], axis=0)
        # else:
        #     x1 = self.data[index-1:index+2]
        
        x1 = self.data[index]
        
        return x1.flatten(), self.logits_1[int(index/self.seg_per_clip)], self.logits_2[int(index/self.seg_per_clip)]
        
        
        