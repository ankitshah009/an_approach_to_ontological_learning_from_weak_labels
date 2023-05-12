import torch
import torch.nn as nn
from pdb import set_trace as bp
import math

class BaseNet(nn.Module):
    
    def __init__(self, in_features):
        super(BaseNet, self).__init__()
        self.out_features = 128
        
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
        self.linear4 = nn.Linear(512, self.out_features)
        self.bn4 = nn.BatchNorm1d(self.out_features) 
    
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

# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.gcn_linear = nn.Linear(in_features, out_features, bias=bias)
#         self.bn = nn.BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, inp, adj):
#         support = self.gcn_linear(inp)
#         support = self.dropout(self.relu(self.bn(support)))
#         outp = torch.matmul(adj, support)
#         # bp()
#         return outp
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.node_embedding_layers = nn.Sequential(
            nn.Linear(in_features, int((in_features + out_features) / 2), bias=bias),
            nn.BatchNorm1d(int((in_features + out_features) / 2)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.Linear(int((in_features + out_features) / 2), out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5)
        )

    def forward(self, inp, adj):
        support = self.node_embedding_layers(inp)
        outp = torch.matmul(adj, support)
        return outp

class SiameseNet_GCN(nn.Module):
    def __init__(self, siamese_in_channel, gcn_in_channel, gcn_adj_matrix):
        super(SiameseNet_GCN, self).__init__()
        
        self.base = BaseNet(siamese_in_channel)
        
        self.gcn1 = GraphConvolution(gcn_in_channel, 512)
        self.gcn1_relu = nn.LeakyReLU(0.2)
        self.gcn2 = GraphConvolution(512, self.base.out_features)

        self.gcn_adj_matrix = gcn_adj_matrix
        
    def forward(self, x, class_embeddings):
        x_embeddings = self.base(x)
        
        class_embeddings_out = self.gcn1(class_embeddings, self.gcn_adj_matrix)
        class_embeddings_out = self.gcn1_relu(class_embeddings_out)
        class_embeddings_out = self.gcn2(class_embeddings_out, self.gcn_adj_matrix)

        class_embeddings_out = class_embeddings_out.transpose(0, 1)
        x_class_cosine_sim = torch.matmul(x_embeddings, class_embeddings_out)
        
        return x_embeddings, x_class_cosine_sim