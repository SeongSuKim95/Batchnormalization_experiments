import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.tensorboard import SummaryWriter
#####################################################

# Basic block class of Resnet18
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, with_BN, stride=1):
        super(BasicBlock, self).__init__()
        self.with_BN = with_BN

        # 3x3 conv filter, pad = 1 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_BN: # batch normalization layer
            self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 conv filter, pad = 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if with_BN: # batch normalization layer
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()# 단순한 identity mapping인 경우

        if stride != 1: # stride가 1이 아니라면, Identity mapping이 아닌 경우
            modules = [nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)]
            if with_BN:
                modules.append(nn.BatchNorm2d(planes))
            self.shortcut = nn.Sequential(*modules)

    def forward(self, x):
        if self.with_BN: # 배치 정규화(batch normalization)를 사용하는 경우
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x) # (핵심) skip connection
        out = F.relu(out)
        return out

# Resnet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, with_BN, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64 ## in_planes -> 64
        self.with_BN = with_BN

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if with_BN: # using batch normalization
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, with_BN, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, with_BN, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, with_BN, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, with_BN, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, with_BN, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) ## strides = [stride, 1, 1, ...] # layers
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, with_BN, stride)) ## block = BasicBlock(self, in_planes, planes, with_BN, stride=1):
            self.in_planes = planes # (in_planes, planes) -> ((in_planes = planes), planes)) 다음 레이어에 연결하기 위해 채널 수 맞추기
        return nn.Sequential(*layers) ## nn.Sequential(*args:Any)

    def forward(self, x):
        if self.with_BN: # 배치 정규화(batch normalization)를 사용하는 경우
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# ResNet18 function
def ResNet18(with_BN): # num_blocks = [2,2,2,2]
    return ResNet(BasicBlock, [2, 2, 2, 2], with_BN)

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# examples = iter(train_loader)
# example_data, example_targets = examples.next()
# #print(example_data.shape) --> [128,3,32,32]
# #matplotlib --> RGB
# #opencv --> BGR
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap = 'gray')
# #Show image batch in matplotlib
# #plt.show()

# # Show one batch of data in tensorboard
# img_grid = torchvision.utils.make_grid(example_data) # should be cpu tensor
# writer.add_image('CIFAR10_images', img_grid)
# #writer_BN.add_image('CIFAR10_images',img_grid)

criterion = nn.CrossEntropyLoss()  # Classification loss

def train(net, optimizer, epoch, step,with_BN):
    net.train()
    # def train(self, mode=True):
    #     r"""Sets the module in training mode."""
    #     self.training = mode
    #     for module in self.children():
    #         module.train(mode)
    #     return self
    correct = 0  # images which predicted well
    total = 0  # total # of image
    steps = []  # Get list of step to plot in matplotlib
    losses = []  # Loss of step

    for _, (inputs, targets) in enumerate(train_loader): ##
        inputs, targets = inputs.to(device), targets.to(device)
        #print(inputs.shape,targets.shape) ## inputs = [128,3,32,32] , targets = [128]
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs.shape) ## outputs = [128,10] -> Batch, class
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1) ## outputs.max(1)= torch.max(output,1) Dim = 1
        #print(predicted.shape) ## predicted = [128]
        #print(predicted)
        correct += predicted.eq(targets).sum().item() #prediction ->> torch.tensor([128]), targets ->> torch.tensor([128])
        total += targets.size(0) # Batch.size

        steps.append(step) ##[0,1,2,3,4,...]
        losses.append(loss.item()) ## Step-wise losses
        step += 1
        if with_BN:
            writer.add_scalar('Training loss::step',losses[-1],step)
        else:
            writer.add_scalar('Training loss::step',losses[-1],step)

    return correct / total, steps, losses

def test(net, optimizer, epoch):
    net.eval()
    # def eval(self):
    #     r"""Sets the module in evaluation mode."""
    #     return self.train(False)
    correct = 0  # 정답을 맞힌 이미지 개수
    total = 0  # 전체 이미지 개수
    loss = 0  # 손실(loss)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return correct / total, loss

print('Initializing model parameter...')
writer = SummaryWriter('./BN_resnet/runs/without_BN')

net = ResNet18(with_BN=False).to(device)
#net = nn.DataParallel(net,device_ids=) ## For multi-gpu using

learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002) ##optim.SGD(net.parameters(),lr,momentum,weight_decay)

# for p in net.parameters():
#      if p.requires_grad:
#          print(p)
#          print(p.numel())

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('# of learnable parameters:', total_params)
#net.parameters() --> learnable parameters of a model

#Initializing

#For plot
without_BN_train_accuracies = []
without_BN_steps = []
without_BN_train_losses = []
without_BN_test_accuracies = []
without_BN_test_losses = []

epochs = 20

for epoch in range(0, epochs):
    print(f'[ Epoch: {epoch}/{epochs} ]')
    train_accuracy, steps, train_losses = train(net, optimizer, epoch, len(without_BN_steps),False) # train return : correct/total , steps, losses --> step wise
    #print(train_accuracy,steps,train_losses)
    ## for ploting
    without_BN_train_accuracies.append(train_accuracy)
    #print(len(without_BN_train_accuracies))
    without_BN_steps.extend(steps)
    #print(len(without_BN_steps))
    without_BN_train_losses.extend(train_losses)
    #print(len(without_BN_train_losses))

    print(f'Train accuracy = {train_accuracy * 100:.2f} / Train loss = {sum(train_losses)}')

    writer.add_scalar('Accuarcy:: epoch',train_accuracy*100, epoch) # y: Accuracy, x: epoch
    writer.add_scalar('Training loss:: epoch', train_losses[-1], epoch)

    test_accuracy, test_loss = test(net, optimizer, epoch)
    without_BN_test_accuracies.append(test_accuracy)
    without_BN_test_losses.append(test_loss)

    print(f'Test accuracy = {test_accuracy * 100:.2f} / Test loss = {test_loss}')
    writer.add_scalar('test_accuarcy:: epoch',test_accuracy*100, epoch) # y: Accuracy, x: epoch
    writer.add_scalar('Test loss:: epoch',test_loss, epoch)

writer.close()
#Training loss / step plot
plt.plot(without_BN_steps, without_BN_train_losses)
plt.title('Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/K_Trainingloss_step_woBN.png")
plt.cla()
#--43
#plt.show()
#plt.savefig("/home/sungsu_dp/PycharmProjects/Batchnorm_Resnet/Result/Trainingloss_step_woBN.png") -- 41

#Test_loss / epoch plot
plt.plot([i for i in range(len(without_BN_test_losses))], without_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/K_Testloss_epoch_woBN.png")
plt.cla()

#Test accuracy / epoch plot
plt.plot([i for i in range(len(without_BN_test_accuracies))], without_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Testaccuracy_epoch_woBN.png")
plt.cla()
# With Batch normalization

print('Initializing model parameters.')

writer = SummaryWriter('./BN_resnet/runs/with_BN')

net = ResNet18(with_BN=True).cuda()
learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('# of learnable parameters:', total_params)

with_BN_train_accuracies = []
with_BN_steps = []
with_BN_train_losses = []
with_BN_test_accuracies = []
with_BN_test_losses = []

epochs = 20

for epoch in range(0, epochs):
    print(f'[ Epoch: {epoch}/{epochs} ]')
    train_accuracy, steps, train_losses = train(net, optimizer, epoch, len(with_BN_steps),True)
    with_BN_train_accuracies.append(train_accuracy)
    with_BN_steps.extend(steps)
    with_BN_train_losses.extend(train_losses)
    print(f'Train accuracy = {train_accuracy * 100:.2f} / Train loss = {sum(train_losses)}')

    #Tensorboard
    writer.add_scalar('Accuarcy:: epoch',train_accuracy*100, epoch) # y: Accuracy, x: epoch
    writer.add_scalar('Training loss:: epoch', train_losses[-1], epoch)

    test_accuracy, test_loss = test(net, optimizer, epoch)

    with_BN_test_accuracies.append(test_accuracy)
    with_BN_test_losses.append(test_loss)

    print(f'Test accuracy = {test_accuracy * 100:.2f} / Test loss = {test_loss}')
    writer.add_scalar('test_accuarcy:: epoch',test_accuracy*100, epoch) # y: Accuracy, x: epoch
    writer.add_scalar('Test loss:: epoch',test_loss, epoch)

writer.close()

# Training_loss/ step plot
plt.plot(with_BN_steps, with_BN_train_losses)
plt.title('Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Trainloss_steps_wBN.png")
plt.cla()

# Test_accuracy / epoch plot
plt.plot([i for i in range(len(with_BN_test_accuracies))], with_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Testaccuracy_epoch_wBN.png")
plt.cla()

# Test_loss / epoch plot
plt.plot([i for i in range(len(with_BN_test_losses))], with_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Testloss_epoch_wBN.png")
plt.cla()

# Train_loss / step plot overlap
plt.plot(without_BN_steps, without_BN_train_losses)
plt.plot(with_BN_steps, with_BN_train_losses)
plt.title('Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Trainloss_step_comparison.png")
plt.cla()

# Test_accuracy / epoch plot overlap
plt.plot([i for i in range(len(without_BN_test_accuracies))], without_BN_test_accuracies)
plt.plot([i for i in range(len(with_BN_test_accuracies))], with_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Testaccuracy_epoch_comparison.png")
plt.cla()

# Test_loss / epoch plot overlap
plt.plot([i for i in range(len(without_BN_test_losses))], without_BN_test_losses)
plt.plot([i for i in range(len(with_BN_test_losses))], with_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/BN_resnet/Result/Testloss_epoch_comparison.png")
plt.cla()
