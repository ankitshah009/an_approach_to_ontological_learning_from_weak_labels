import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

class AudioSet_Siamese(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        # self.data = data.reshape(-1, data.shape[2])
        self.seg_per_clip = 10
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2  
        self.context = 1
        num_clips = len(data)
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        self.same_subclass = []
        self.diff_subclass_same_superclass = []
        self.diff_superclass = []
            
        
        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i].astype('int')] = 1 
            self.logits_2[i][self.labels2[i].astype('int')] = 1
        
        self.same_subclass = [np.where( np.all(self.logits_1[i] == self.logits_1, axis=1) )[0] for i in range(num_clips)]
        self.diff_subclass_same_superclass = [np.where( np.logical_and( np.logical_not( np.all(self.logits_1[i] == self.logits_1, axis=1)),  
                                                                       np.all(self.logits_2[i] == self.logits_2, axis=1)) )[0] for i in range(num_clips)]
        self.diff_superclass = [ np.where( np.logical_not( np.all(self.logits_2[i] == self.logits_2, axis=1) ) )[0] for i in range(num_clips) ]
        
    def __len__(self):
        
        return int(self.length * 5)
        
    def __getitem__(self, idx):
        
        idx = np.random.choice(np.arange(self.length))
        class_idx = int(idx/self.seg_per_clip)
        class1_1 = self.labels1[class_idx]
        class1_2 = self.labels2[class_idx]
        
        class2_type = random.uniform(0, 1)
        
        if(class2_type < 1/3):
            
            # Same subclass
            x2_idxs = self.same_subclass[class_idx]             
            x2_idx = np.random.choice(x2_idxs) 
            pair_type = 0
        
        elif (class2_type < 2/3):
        
            # Same superclass, different subclass
            x2_idxs = self.diff_subclass_same_superclass[class_idx]   
            if x2_idxs.size != 0:
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 1
            else:
                x2_idxs = self.diff_superclass[class_idx]
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 2
                
        else:
            
            # Different superclass
            x2_idxs = self.diff_superclass[class_idx]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 2
            
        
        # if (idx % 10 == 0):
        #     x1 = np.repeat(self.data[idx:idx+2], [2, 1], axis=0) 
        # elif (idx % 10 == 9):
        #     x1 = np.repeat(self.data[idx-1:idx+1], [1, 2], axis=0)
        # else:
        #     x1 = self.data[idx-1:idx+2]
        
        x1 = self.data[idx]
        
        # Random second data point
        x2_train_idx = x2_idx * self.seg_per_clip + np.random.choice(np.arange(10))
        # if (x2_train_idx % 10 == 0):
        #     x2 = np.repeat(self.data[x2_train_idx:x2_train_idx+2], [2, 1], axis=0) 
        # elif (x2_train_idx % 10 == 9):
        #     x2 = np.repeat(self.data[x2_train_idx-1:x2_train_idx+1], [1, 2], axis=0)
        # else:
        #     x2 = self.data[x2_train_idx-1:x2_train_idx+2]

        
        x2 = self.data[x2_train_idx]
        
        # Superclass
        x1_superclass = self.logits_2[class_idx]
        x2_superclass = self.logits_2[x2_idx]
        
        # Subclass
        x1_subclass = self.logits_1[class_idx]
        x2_subclass = self.logits_1[x2_idx]    
                           
        return x1.flatten(), x2.flatten(), x1_subclass, x1_superclass, x2_subclass, x2_superclass, pair_type        

    
class AudioSet_Siamese_Eval(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        # self.data = data.reshape(-1, data.shape[2])
        self.seg_per_clip = 10
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
        
        
class AudioSet_Strong_Siamese(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        # self.data = data.reshape(-1, data.shape[2])
        self.seg_per_clip = 10
        self.length = self.data.shape[0]
        # self.labels1 = np.concatenate(labels1, axis=0)
        self.labels1 = labels1
        # print(len(data))
        # print(len(labels1))
        if len(data) == len(labels1):
            self.labels1 = np.concatenate(labels1, axis=0)
        
        self.labels1 = [np.asarray(lbl) for lbl in self.labels1]
        print(self.labels1[0])
        
        # self.labels2 = np.concatenate(labels2, axis=0)     
        self.labels2 = labels2
        if len(data) == len(labels2):
            self.labels2 = np.concatenate(labels2, axis=0)
            # print('hi')
        
        
        self.labels2 = [np.asarray(lbl) for lbl in self.labels2]
        self.context = 1
        num_clips = len(self.data)
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        self.same_subclass = []
        self.diff_subclass_same_superclass = []
        self.diff_superclass = []
            
        
        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            # print(self.labels1[i].astype('int'))
            # print(self.labels1[i])
            self.logits_1[i][self.labels1[i]] = 1 
            self.logits_2[i][self.labels2[i]] = 1
        
        # self.same_subclass = [np.where( np.all(self.logits_1[i] == self.logits_1, axis=1) )[0] for i in range(num_clips)]
        # self.diff_subclass_same_superclass = [np.where( np.logical_and( np.logical_not( np.all(self.logits_1[i] == self.logits_1, axis=1)),  
        #                                                                np.all(self.logits_2[i] == self.logits_2, axis=1)) )[0] for i in range(num_clips)]
        # self.diff_superclass = [ np.where( np.logical_not( np.all(self.logits_2[i] == self.logits_2, axis=1) ) )[0] for i in range(num_clips) ]
        
    def __len__(self):
        
        return int(self.length * 5)
        
    def __getitem__(self, idx):
        
        idx = np.random.choice(np.arange(self.length))
        # class_idx = int(idx/self.seg_per_clip)
        class1_1 = self.labels1[idx]
        class1_2 = self.labels2[idx]
        
        class2_type = random.uniform(0, 1)
        
        if(class2_type < 1/3):
            
            # Same subclass
            x2_idxs = np.where( np.logical_and( np.all(self.logits_1[idx] == self.logits_1, axis=1), np.all(self.logits_2[idx] == self.logits_2, axis=1) ))[0]
            # x2_idxs = self.same_subclass[class_idx]             
            x2_idx = np.random.choice(x2_idxs) 
            pair_type = 0
        
        elif (class2_type < 2/3):
        
            # Same superclass, different subclass
            # x2_idxs = self.diff_subclass_same_superclass[class_idx]   
            # x2_idxs = np.where( np.logical_and( np.logical_not( np.all(self.logits_1[idx] == self.logits_1, axis=1)),  
            #                                                            np.all(self.logits_2[idx] == self.logits_2, axis=1)) )[0]
            
            x2_idxs = np.where(np.logical_and( np.sum(self.logits_1[idx] != self.logits_1, axis=1) == np.sum(self.logits_1[idx]) + np.sum(self.logits_1, axis=1), 
                                                                        np.all(self.logits_2[idx] == self.logits_2, axis=1)) )[0]
            
            if x2_idxs.size != 0:
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 1
            else:
                x2_idxs = np.where( np.logical_not( np.all(self.logits_2[idx] == self.logits_2, axis=1) ) )[0]
                # x2_idxs = self.diff_superclass[class_idx]
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 2
                
        else:
            
            # Different superclass
            x2_idxs = np.where( np.logical_not( np.all(self.logits_2[idx] == self.logits_2, axis=1) ) )[0]
            # x2_idxs = self.diff_superclass[class_idx]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 2
            
        
        # if (idx % 10 == 0):
        #     x1 = np.repeat(self.data[idx:idx+2], [2, 1], axis=0) 
        # elif (idx % 10 == 9):
        #     x1 = np.repeat(self.data[idx-1:idx+1], [1, 2], axis=0)
        # else:
        #     x1 = self.data[idx-1:idx+2]
        
        x1 = self.data[idx]
        
        # Random second data point
        # x2_train_idx = x2_idx * self.seg_per_clip + np.random.choice(np.arange(10))
        # if (x2_train_idx % 10 == 0):
        #     x2 = np.repeat(self.data[x2_train_idx:x2_train_idx+2], [2, 1], axis=0) 
        # elif (x2_train_idx % 10 == 9):
        #     x2 = np.repeat(self.data[x2_train_idx-1:x2_train_idx+1], [1, 2], axis=0)
        # else:
        #     x2 = self.data[x2_train_idx-1:x2_train_idx+2]

        
        x2 = self.data[x2_idx]
        
        # Superclass
        x1_superclass = self.logits_2[idx]
        x2_superclass = self.logits_2[x2_idx]
        
        # Subclass
        x1_subclass = self.logits_1[idx]
        x2_subclass = self.logits_1[x2_idx]    
        
        # print(x1_subclass)
        # print(x2_subclass)
        # print(pair_type)
                           
        return x1.flatten(), x2.flatten(), x1_subclass, x1_superclass, x2_subclass, x2_superclass, pair_type        

class AudioSet_Strong_Siamese_Eval(Dataset):
    
    def __init__(self, data, labels1, labels2, num_subclasses, num_superclasses):
        
        self.data = np.concatenate(data, axis=0).astype('float')
        self.seg_per_clip = 10
        self.length = self.data.shape[0]
        self.labels1 = np.concatenate(labels1, axis=0)
        # self.labels1 = labels1
        self.labels1 = [np.asarray(lbl) for lbl in self.labels1]
        self.labels2 = np.concatenate(labels2, axis=0)       
        # self.labels2 = labels2
        self.labels2 = [np.asarray(lbl) for lbl in self.labels2]
        
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses
        num_clips = self.data.shape[0]

        self.logits_1 = np.zeros((num_clips, num_subclasses)).astype('long')
        self.logits_2 = np.zeros((num_clips, num_superclasses)).astype('long')
        
        for i in range(num_clips):
            self.logits_1[i][self.labels1[i]] = 1 
            self.logits_2[i][self.labels2[i]] = 1
        
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
        
        return x1.flatten(), self.logits_1[index], self.logits_2[index]