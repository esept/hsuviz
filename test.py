import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib as mp
import numpy as np
from hsuviz.hsuviz import hsuviz

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,9,3,padding = 2)
        self.maxp1 = nn.MaxPool2d(3,stride = 1)
        self.conv2 = nn.Conv2d(9,5,5)
        self.conv3 = nn.Conv2d(5,1,3)
        self.__list = [self.conv1,self.maxp1,self.conv2,self.conv3]
        # self.newX = []
        # pass
    def forward(self,input):
        self.__input = input
        x = self.conv1(input)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    def get_layers(self):
        return self.__list

    
if __name__ == "__main__":
    model = Net()
    vv = hsuviz(model,[1,218,218])
    # print(model.modules())
    # vv.draw(name="new")
    print(vv.types)
    for i in vv.types:
        ss = i.state_dict()
        if len(ss) != 0 :
            print("\n-------------------------------------------------------------------------------------")
            print(ss['weight'].shape)
            print(ss['bias'].shape)