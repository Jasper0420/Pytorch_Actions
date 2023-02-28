import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model_standard import AlexNet
#这是封装版本的训练代码，这样的代码方便相关模块即插即用，可读性比较强
# 定义训练参数
batch_size = 64
num_epochs = 10
learning_rate = 0.01
#加载数据集
def load_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader

def build_model():
    net = AlexNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    return net, loss_fn, optimizer

def train(net, loss_fn, optimizer, trainloader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)

            # 计算损失函数
            loss = loss_fn(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失值
            running_loss += loss.item()
            if i % 200 == 199:    # 每 200 个小批量数据打印一次训练状态
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            #torch.max() 函数返回一个元组，包含两个张量：最大值和最大值对应的索引。
            #我们使用“_”来表示我们不需要使用第一个张量（最大值），只需要使用第二个张量（即最大值对应的索引）。
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    trainloader, testloader = load_data(batch_size)
    net, loss_fn, optimizer = build_model()
    train(net, loss_fn, optimizer, trainloader, num_epochs)
    test(net, testloader)
