import numpy as np
import random
from torch.utils.data import DataLoader, Dataset


class MSoSSiamese(Dataset):
    def __init__(self, data, labels1, labels2, num_subclasses=97, num_superclasses=5) -> None:
        super().__init__()

        self.data = np.concatenate(data, axis=0).astype('float')
        self.length = self.data.shape[0]
        self.data_len = data.shape[1]
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
            self.logits_1[i][self.labels1[i]] = 1
            self.logits_2[i][self.labels2[i]] = 1

        self.same_subclass = [np.where(np.all(self.logits_1[i] == self.logits_1, axis=1))[
            0] for i in range(num_clips)]
        self.diff_subclass_same_superclass = [np.where(np.logical_and(np.logical_not(np.all(self.logits_1[i] == self.logits_1, axis=1)),
                                                                      np.all(self.logits_2[i] == self.logits_2, axis=1)))[0] for i in range(num_clips)]
        self.diff_superclass = [np.where(np.logical_not(np.all(
            self.logits_2[i] == self.logits_2, axis=1)))[0] for i in range(num_clips)]

    def __len__(self):

        return int(self.length * self.data_len)

    def __getitem__(self, idx):

        idx = np.random.choice(np.arange(self.length))
        class_idx = int(idx / self.data_len)
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
        x2_train_idx = x2_idx * 10 + np.random.choice(np.arange(10))
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


class MSoSSiameseEval(Dataset):

    def __init__(self, data, labels1, labels2, num_subclasses=97, num_superclasses=5):

        self.data = np.concatenate(data, axis=0).astype('float')
        self.length = self.data.shape[0]
        self.data_len = data.shape[1]
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

        return x1.flatten(), self.logits_1[int(index / self.data_len)], self.logits_2[int(index / self.data_len)]
