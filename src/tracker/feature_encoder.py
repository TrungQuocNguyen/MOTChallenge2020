import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class NodeEncoder(nn.Module): 
    def __init__(self):
        super(NodeEncoder,self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size = (3,3), stride = 1, padding = 1, bias = False)
        self.resnet50.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size = (1,1), stride = 1, bias = False)
        self.resnet50.fc = nn.Linear(in_features = 2048, out_features = 512, bias = True)
        
        self.fc2 = nn.Linear(in_features = 512, out_features = 128, bias = True)
        self.fc3 = nn.Linear(in_features = 128, out_features = 32, bias = True)
    def forward(self, x):
        x = self.resnet50(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class EdgeEncoder(nn.Module): 
    def __init__(self):
        super(EdgeEncoder,self).__init__()
        self.fc1 = nn.Linear(in_features = 6, out_features = 18, bias = True)
        self.fc2 = nn.Linear(in_features = 18, out_features = 18, bias = True)
        self.fc3 = nn.Linear(in_features = 18, out_features = 16, bias = True)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x