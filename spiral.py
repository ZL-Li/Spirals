# spiral.py
# python 3.7.7
# torch 1.6.0
# torchversion 0.7.0

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # input layer(2) -> hidden layer(num_hid) -> output layer(1)
        self.in_to_hidden = nn.Linear(2, num_hid)
        self.hidden_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        x, y = input[:, 0], input[:, 1]
        r = torch.sqrt(x * x + y* y) # r is a scalar
        a = torch.atan2(y, x) # a is a scalar
        r = r.view(-1, 1) # change r to a column
        a = a.view(-1, 1) # change a to a column
        ans = torch.cat((r, a), dim = 1) # cat r and a to 2 columns
        self.hidden1_output = torch.tanh(self.in_to_hidden(ans))
        output = torch.sigmoid(self.hidden_to_out(self.hidden1_output))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # input layer(2) -> hidden layer1(num_hid) -> hidden layer2(num_hid) -> output layer(1)
        self.in_to_hidden1 = nn.Linear(2, num_hid)
        self.hidden1_to_hidden2 = nn.Linear(num_hid, num_hid)
        self.hidden2_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        self.hidden1_output = torch.tanh(self.in_to_hidden1(input))
        self.hidden2_output = torch.tanh(self.hidden1_to_hidden2(self.hidden1_output))
        output = torch.sigmoid(self.hidden2_to_out(self.hidden2_output))
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again
        
        if layer == 1: # the first hidden layer of PolarNet and RawNet
            pred = (net.hidden1_output[:, node] >= 0).float()
        else: # layer == 2, the second hidden layer of PolarNet
            pred = (net.hidden2_output[:, node] >= 0).float()
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
