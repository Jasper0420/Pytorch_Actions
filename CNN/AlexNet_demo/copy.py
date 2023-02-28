import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model_standard import AlexNet

# 定义训练参数
batch_size = 64 
num_epochs = 10
learning_rate = 0.01

def load_data(batch_size):
    #数据集的预处理
    #transforms.ToTensor()将图像数据从PIL格式转换为PyTorch张量
    #使用transforms.Normalize()将图像数据标准化。
    transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transforms) 
    trainloader = torch.utils.data.Dataloader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms)
    testloader = torch.utils.data.Dataloader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def build_model():
    net = AlexNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nn.parameters(), lr=learning_rate, momentum=0.9)
    return net, loss_fn, optimizer

def train(net, loss_fn, optimizer, trainloader, num_epochs):
    for epoch in range(num_epochs):
        running_rate = 0.0
        for i, data in enumerate(trainloader, 0):
            input, labels = data
