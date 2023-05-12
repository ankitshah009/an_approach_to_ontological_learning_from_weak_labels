import torch
import torch.nn as nn

import math

class BaseNet(nn.Module):
    
    def __init__(self, in_features):
        super(BaseNet, self).__init__()
        
        # self.layers = nn.Sequential(nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())  
        
        self.linear1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.linear4 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128) 
    
    def forward(self, x):
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.linear4(out)
        out = self.bn4(out)
        
        return out
    
class SiameseNet_OntologicalLayer(nn.Module):
    
    def __init__(self, num_classes_2, num_classes_1, in_features, M):
        super(SiameseNet_OntologicalLayer, self).__init__()
        num_classes = num_classes_1 + num_classes_2
        
        self.base = BaseNet(in_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        embedding = self.base(x)
        
        out = self.bn(self.output_layer(self.dropout(self.relu(embedding))))
        
        # out_prob = self.sigmoid(out)
        
        return embedding, out