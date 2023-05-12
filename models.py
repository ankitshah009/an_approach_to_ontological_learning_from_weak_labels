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
    
class BaseNet_1(nn.Module):
    
    def __init__(self, in_features):
        super(BaseNet_1, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.Dropout(0.3), nn.ReLU(),
                                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Dropout(0.3), nn.ReLU(),
                                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(0.3), nn.ReLU(),
                                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(0.3))  
    
    def forward(self, x):
        
        return self.layers(x)
    
    
class SiameseNet_OntologicalLayer(nn.Module):
    
    def __init__(self, num_classes_2, num_classes_1, in_features, M):
        super(SiameseNet_OntologicalLayer, self).__init__()
        
        self.base = BaseNet(in_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, num_classes_1)
        self.bn = nn.BatchNorm1d(num_classes_1)

        self.sigmoid = nn.Sigmoid()

        self.ontological_layer = nn.Linear(num_classes_1, num_classes_2, bias=False)
        self.ontological_layer.weight.requires_grad = False
        
        with torch.no_grad():
            self.ontological_layer.weight.copy_(torch.from_numpy(M))
        
    def forward(self, x1, x2):
        
        embedding1 = self.base(x1)
        embedding2 = self.base(x2)
        
        out1_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding1))))
        out2_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding2))))
        
        out1_1_prob = self.sigmoid(out1_1)
        out2_1_prob = self.sigmoid(out2_1)
        
        out1_2 = self.ontological_layer(out1_1_prob)
        out2_2 = self.ontological_layer(out2_1_prob)      
        
        return embedding1, embedding2, out1_1, out1_2, out2_1, out2_2


class YamnetBaseNet(nn.Module):
    
    def __init__(self, in_features):
        super(YamnetBaseNet, self).__init__()
        
        # self.layers = nn.Sequential(nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), 
        #                             nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())  
        
        self.linear1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64) 
    
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
    
class YamnetSiameseNet_OntologicalLayer(nn.Module):
    
    def __init__(self, num_classes_2, num_classes_1, in_features, M):
        super(YamnetSiameseNet_OntologicalLayer, self).__init__()
        
        self.base = YamnetBaseNet(in_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, num_classes_1)
        self.bn = nn.BatchNorm1d(num_classes_1)

        self.sigmoid = nn.Sigmoid()

        self.ontological_layer = nn.Linear(num_classes_1, num_classes_2, bias=False)
        self.ontological_layer.weight.requires_grad = False
        
        with torch.no_grad():
            self.ontological_layer.weight.copy_(torch.from_numpy(M))
        
    def forward(self, x1, x2):
        
        embedding1 = self.base(x1)
        embedding2 = self.base(x2)
        
        out1_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding1))))
        out2_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding2))))
        
        out1_1_prob = self.sigmoid(out1_1)
        out2_1_prob = self.sigmoid(out2_1)
        
        out1_2 = self.ontological_layer(out1_1_prob)
        out2_2 = self.ontological_layer(out2_1_prob)      
        
        return embedding1, embedding2, out1_1, out1_2, out2_1, out2_2

    

    
    