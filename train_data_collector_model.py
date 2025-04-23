# data_collector_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataCollectorCNN(nn.Module):
    def __init__(self, input_channels=1, action_dim=5):
        super(DataCollectorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(6, 24, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 96, 96)
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(sample_input)))))
            self.fc_input_dim = x.numel() // x.size(0)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x