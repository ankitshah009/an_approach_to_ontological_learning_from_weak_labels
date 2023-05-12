import torch
import torch.nn as nn

class Ontological_Loss(nn.Module):
    
    def __init__(self, lambda1, lambda2, lambda3, weights_1, weights_2):
        super(Ontological_Loss, self).__init__()
        
        self.ontology_targets = [0, 5, 10]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.weights_1 = weights_1
        self.weights_2 = weights_2
      
    def forward(self, out, targets, pair_type):
        
        embedding1, embedding2, out1_1, out1_2, out2_1, out2_2 = out
        target1_1, target1_2, target2_1, target2_2 = targets
        
        ont_target = pair_type
        
        loss1 = nn.BCEWithLogitsLoss(pos_weight=self.weights_1)
        # print(out1_1.type())
        # print(target1_1.type())
        L1_1 = loss1(out1_1, target1_1.float())
        L1_2 = loss1(out2_1, target2_1.float())
        
        # loss2 = nn.BCELoss(weight=self.weights_2)
        # loss2 = nn.BCELoss()
        loss2 = nn.BCEWithLogitsLoss(pos_weight=self.weights_2)
        out1_2 = -torch.log((1 / out1_2) - 1)
        out2_2 = -torch.log((1 / out2_2) - 1)

        L2_1 = loss2(out1_2, target1_2.float())
        L2_2 = loss2(out2_2, target2_2.float())
        
        D_w = (torch.sqrt(torch.sum((embedding1 - embedding2)**2, axis=1)) - ont_target)**2
        
        loss = self.lambda1 * (L1_1 + L1_2) + self.lambda2 * (L2_1 + L2_2) + self.lambda3 * D_w
        
        return loss.mean()
        
        
class Ontological_Loss_Unweighted(nn.Module):

    def __init__(self, lambda1, lambda2, lambda3):
        super(Ontological_Loss_Unweighted, self).__init__()

        self.ontology_targets = [0, 5, 10]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        # self.weights_1 = weights_1
        # self.weights_2 = weights_2

    def forward(self, out, targets, pair_type):

        embedding1, embedding2, out1_1, out1_2, out2_1, out2_2 = out
        target1_1, target1_2, target2_1, target2_2 = targets

        ont_target = pair_type * 2

        loss1 = nn.BCEWithLogitsLoss()
        # print(out1_1.type())
        # print(target1_1.type())
        L1_1 = loss1(out1_1, target1_1.float())
        L1_2 = loss1(out2_1, target2_1.float())

        loss2 = nn.BCELoss()
        L2_1 = loss2(out1_2, target1_2.float())
        L2_2 = loss2(out2_2, target2_2.float())

        D_w = (torch.sqrt(torch.sum((embedding1 - embedding2)**2, axis=1)) - ont_target)**2

        loss = self.lambda1 * (L1_1 + L1_2) + self.lambda2 * (L2_1 + L2_2) + self.lambda3 * D_w

        return loss.mean()
