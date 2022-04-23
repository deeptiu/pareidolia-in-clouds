import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset


# Pre-trained weights up to second-to-last layer
# final layers should be initialized from scratch!
class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ResNet = models.resnet18(pretrained=True)
        out_features = 20
        self.ResNet.fc = nn.Linear(self.ResNet.fc.in_features, out_features)
        self.ResNet.fc.weight.data = nn.init.xavier_normal_(torch.ones_like(self.ResNet.fc.weight.data))
        self.ResNet.fc.bias.data = torch.zeros_like(self.ResNet.fc.bias.data)
    
    def forward(self, x):
        return self.ResNet(x)

args = ARGS()
args.use_cuda = True
args.epochs = 10
args.batch_size = 32
args.lr = 1e-4
args.log_every = 100
args.save_at_end = True
args.save_freq = 5
args.step_size = 20
args.gamma = 0.4
print(args)

model = PretrainedResNet()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
test_ap, test_map = trainer.train(args, model, optimizer, scheduler, model_name='cloud')
