import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.dataloader import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Represent output channels after Conv layer
VGG_types = {
    'VGG_SS': [64,'M', 128, 'M', 256, 256, 'M', 512, 512 , 'M', 512, 512], 
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512 , 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):  
    def __init__(self,with_BN,in_channels = 3,num_classes = 10):
        super(VGG_net,self).__init__()
        self.with_BN = with_BN
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG_types['VGG11'])
        self.fcs = nn.Sequential(
                    nn.Linear(512*7*7,4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096,2048),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(2048, num_classes)
                    )

    def forward(self,x):

        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        
        x = self.fcs(x)

        return x

    def create_conv_layer(self,architecture):

        layers = []

        in_channels = self.in_channels
         
        ## Simliar method to ResNet
        for x in architecture:

            if type(x) == int: # Conv layers
                out_channels = x
                
                if self.with_BN:

                    layers += [nn.Conv2d(in_channels = in_channels,out_channels = out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU(),]
                    in_channels = x 
                    
                else :

                    layers += [nn.Conv2d(in_channels = in_channels,out_channels = out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                            nn.ReLU(),]
                    in_channels = x 

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
            
        return nn.Sequential(*layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x = torch.randn(16,3,224,224).to(device)
# print(model(x))
# Data
transform_train = transforms.Compose([
    transforms.Resize((224,224),interpolation = 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224),interpolation = 2),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='/home/sungsu21/data/', train=True, download=False, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='/home/sungsu21/data/', train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()  # Classification loss

def train(net, optimizer, epoch, step, with_BN):
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

writer = SummaryWriter('./runs/Vggnet/without_BN')

net = VGG_net(with_BN = False).to(device)
#net = nn.DataParallel(net,device_ids=) ## For multi-gpu using

learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002) ##optim.SGD(net.parameters(),lr,momentum,weight_decay)

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

epochs = 50

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
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Trainingloss_step_woBN.png")
plt.cla()
#--43
#plt.show()
#plt.savefig("/home/sungsu_dp/PycharmProjects/Batchnorm_Resnet/Result/Trainingloss_step_woBN.png") -- 41

#Test_loss / epoch plot
plt.plot([i for i in range(len(without_BN_test_losses))], without_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testloss_epoch_woBN.png")
plt.cla()

#Test accuracy / epoch plot
plt.plot([i for i in range(len(without_BN_test_accuracies))], without_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testaccuracy_epoch_woBN.png")
plt.cla()
# With Batch normalization

print('Initializing model parameters.')

writer = SummaryWriter('./runs/Vggnet/with_BN')

net = VGG_net(with_BN=True).to(device)
learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('# of learnable parameters:', total_params)

with_BN_train_accuracies = []
with_BN_steps = []
with_BN_train_losses = []
with_BN_test_accuracies = []
with_BN_test_losses = []

epochs = 50

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
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Trainloss_steps_wBN.png")
plt.cla()

# Test_accuracy / epoch plot
plt.plot([i for i in range(len(with_BN_test_accuracies))], with_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testaccuracy_epoch_wBN.png")
plt.cla()

# Test_loss / epoch plot
plt.plot([i for i in range(len(with_BN_test_losses))], with_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testloss_epoch_wBN.png")
plt.cla()

# Train_loss / step plot overlap
plt.plot(without_BN_steps, without_BN_train_losses)
plt.plot(with_BN_steps, with_BN_train_losses)
plt.title('Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Trainloss_step_comparison.png")
plt.cla()

# Test_accuracy / epoch plot overlap
plt.plot([i for i in range(len(without_BN_test_accuracies))], without_BN_test_accuracies)
plt.plot([i for i in range(len(with_BN_test_accuracies))], with_BN_test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testaccuracy_epoch_comparison.png")
plt.cla()

# Test_loss / epoch plot overlap
plt.plot([i for i in range(len(without_BN_test_losses))], without_BN_test_losses)
plt.plot([i for i in range(len(with_BN_test_losses))], with_BN_test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['without BN', 'with BN'])
#plt.show()
plt.savefig("/home/sungsu21/Project/Batch_Normalization/Result/Vggnet/Testloss_epoch_comparison.png")
plt.cla()
