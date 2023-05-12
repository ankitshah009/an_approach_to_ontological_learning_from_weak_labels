from pdb import set_trace as bp
import torch
import torch.nn as nn

class Ontological_Loss(nn.Module):
    
    def __init__(self, weights_1, weights_2):
        super(Ontological_Loss, self).__init__()
        self.weights = torch.cat((weights_1, weights_2))
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=self.weights)
      
    def forward(self, out, targets):
        
        _, out = out
        
        loss = self.loss_function(out, targets.float())
        
        return loss.mean()
        