import torch
import torch.nn as nn
import torch.nn.functional as F

################# Classification ################

class Classifier(nn.Module):
    def __init__(self, last_channel, num_cls):
        super(Classifier, self).__init__()
        if last_channel > 128:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_cls)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_cls)
            )
    def forward(self, x):
        return self.classifier(x)
    

########################### Segmentation #################################
