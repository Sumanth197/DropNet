import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import sys
import pickle
import matplotlib.font_manager
from matplotlib import rc, rcParams
import copy

from matplotlib import pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train - Valid - Test Loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


num_train = len(train_dataset)
indices = list(range(num_train))
split = 5000

shuffle = True
random_seed = 42
if shuffle == True:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# Create data loaders for batch processing
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           sampler = train_sampler)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          sampler = valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_activations(model, dataloader):
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    layers = [np.empty((0, each)) for each in [layer.out_channels for layer in conv_layers]]
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        _, acts = model(data)
        for i in range(len(acts)):
            layers[i] = np.concatenate((layers[i], acts[i].cpu().numpy()), axis=0)
    
    return layers

def accuracy(network, dataloader):
    # network.eval()
    total_correct = 0
    total_instances = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        output, acts = network(images)
        _, yhat = torch.max(output.data, 1)
        total_correct += (yhat == labels).sum().item()
        total_instances+=len(images)
    return round(total_correct/total_instances, 3)

def add_filter_mask(mask, model, activationarray, layer, initialize=True):
    """Function to add mask to filters in Conv2D
    Inputs:
    mask - mask which contains either 0 (node/filter dropped) or 1 (node/filter remaining)
    model - PyTorch model
    activationarray - list of PyTorch layers for which we care about their activation values
    layer - the current layer number
    initialize - Boolean. True if we want to reset all the masks to 1
    
    Output:
    model - Updated PyTorch model with the mask layer
    activationarray - Updated list of layers for which the activation values are important
    layer - the updated layer number count
    """
    
    global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    output_tensor = global_avg_pool(model)
    model2 = output_tensor.view(output_tensor.size(0), -1)
    
    # only initialize if this is the first time
    if initialize is True:
        mask[layer] = torch.ones(model2.shape[1:]).to(device)
    
   # Multiply the activation with the filters
    model2 = model2 * mask[layer].reshape(model2.shape[1:])
    activationarray.append(model2)
    
    # do the multiply for the original filters using broadcasting
    model = model * mask[layer].reshape(model.shape[1], 1, 1)
    
    # increase layer count for next iteration
    layer = layer + 1
    return model, activationarray, layer

class alexNet(nn.Module):
    def __init__(self, mask = None):
        super(alexNet, self).__init__()
        
        self.layername = {}
        self.layer = 0
        
        if mask is None:
            self.mask = {}
            self.initialize = True
        else:
            self.mask = mask
            self.initialize = False
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 11, stride=4, padding=5)
        if self.initialize is True:
            self.mask[self.layer] = torch.ones(64).to(device)
            self.layer = self.layer + 1
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        if self.initialize is True:
            self.mask[self.layer] = torch.ones(192).to(device)
            self.layer = self.layer + 1
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        if self.initialize is True:
            self.mask[self.layer] = torch.ones(384).to(device)
            self.layer = self.layer + 1
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding = 1)
        if self.initialize is True:
            self.mask[self.layer] = torch.ones(256).to(device)
            self.layer = self.layer + 1
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        if self.initialize is True:
            self.mask[self.layer] = torch.ones(256).to(device)
            self.layer = self.layer + 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
    
    def getMask(self):
        # print(self.mask)
        return self.mask
    
    def forward(self, x):
        
        self.layer = 0
        self.activationarray = []
        
        x = F.relu(self.conv1(x))
        x, self.activationarray, self.layer = add_filter_mask(self.mask, x, 
                                                              self.activationarray, 
                                                              self.layer, initialize=self.initialize)
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x, self.activationarray, self.layer = add_filter_mask(self.mask, x, 
                                                              self.activationarray, 
                                                              self.layer, initialize=self.initialize)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x, self.activationarray, self.layer = add_filter_mask(self.mask, x, 
                                                              self.activationarray, 
                                                              self.layer, initialize=self.initialize)
        x = F.relu(self.conv4(x))
        x, self.activationarray, self.layer = add_filter_mask(self.mask, x, 
                                                              self.activationarray, 
                                                              self.layer, initialize=self.initialize)
        x = F.relu(self.conv5(x))
        x, self.activationarray, self.layer = add_filter_mask(self.mask, x, 
                                                              self.activationarray, 
                                                              self.layer, initialize=self.initialize)
        x = self.pool(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=1)
        
        return out, self.activationarray

def generatemodel(criterion, patience,
                  num_epochs = 100, percentile = 0.2, numtrials = 15,
                  printvalue = True, printmask = False, printactivation = False):
    
    percentremoved ={}
    train_accuracies = {}
    valid_accuracies = {}
    test_accuracies = {}
    train_losses = {}
    valid_losses ={}
    test_losses = {}
    early_stopping = {}
    oraclecomparison = {}
    masktypes = ['min', 'max', 'random', 'min_layer', 'max_layer', 'random_layer']

    for masktype in masktypes:
        percentremoved[masktype] = []
        train_accuracies[masktype] = []
        valid_accuracies[masktype] = []
        test_accuracies[masktype] = []
        train_losses[masktype] = []
        valid_losses[masktype] = []
        test_losses[masktype] = []
        early_stopping[masktype] = []

    for trial in range(numtrials):

        print('>>> Random seed number: ', trial)

        # Initialize the model
        model = alexNet().to(device)
        mask = model.getMask()
        
        opt = optim.SGD(model.parameters(), lr = 0.1)
        # mask, model, activationmodel = initializemodel()

        # weights_initial = []
        # for param in model.parameters():
        #    weights_initial.append(param.data.clone().detach().tolist())
        weights_initial = model.state_dict()
        optimizer_initial = opt.state_dict()
        
        # print(optimizer_initial)

        for masktype in masktypes:

            print('\n>>> Currently doing', masktype, 'mask <<<')

            mask = resetmask(mask)
            percent = percentmask(mask)

            while percent > 0.1:

                # Initialize the model with the new mask
                torch.cuda.empty_cache() # Clear GPU cache if using GPU
                torch.manual_seed(trial) # Set random seed for torch
                np.random.seed(trial) # Set random seed for numpy 

                # _, model, activationmodel = initializemodel(mask)
                model = alexNet(mask).to(device)
                model.load_state_dict(weights_initial)
                opt = optim.SGD(model.parameters(), lr = 0.1)
                # opt.load_state_dict(optimizer_initial)
                
                train_accuracy, valid_accuracy = [], []
                training_loss, validation_loss = [], []
                best_acc = 0.0
                for epoch in range(num_epochs):
                    
                    # train(model, device, train_loader, opt, epoch)
                    # print(model.state_dict())
                    # test(model, device, test_loader)
                    
                    model.train()
                    train_acc, valid_acc = 0.0, 0.0
                    train_loss, val_loss = 0.0, 0.0

                    for i, (inputs, labels) in enumerate(train_loader):
                        # print("Training Loader")
                        inputs, labels = inputs.to(device), labels.to(device)
                        opt.zero_grad()
                        outputs, acts = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        opt.step()
                        train_loss += loss.item()
                        
                        if i % 500 == 0:
                            # print(i)
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, i * len(inputs), len(train_loader.dataset),
                                100. * i / len(train_loader), loss.item()))
                    

                    model.eval()
                    with torch.no_grad():
                        for data, labels in valid_loader:
                            data, labels = data.to(device), labels.to(device)
                            target, acts = model(data)
                            loss = criterion(target, labels)
                            val_loss += loss.item()

                        valid_acc = accuracy(model, valid_loader)
                        train_acc = accuracy(model, train_loader)
                    
                    train_loss = train_loss / len(train_loader)
                    val_loss = val_loss / len(valid_loader)
                    
                    train_accuracy.append(train_acc)
                    valid_accuracy.append(valid_acc)
                    training_loss.append(train_loss)
                    validation_loss.append(val_loss)
                    
                    print("Training Accuracy :", train_acc)
                    print("Validation Accuracy :", valid_acc)
                    
                    
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        es = 0
                        best_weights = model.state_dict()
                        # torch.save(model.state_dict(), "VGG19_model_" + str(epoch) + 'weight.pt')
                    else:
                        es += 1
                        print("Counter {} of 5".format(es))
                    
                    if es > 4:
                        print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", valid_acc, "...")
                        print("*" * 50)
                        break
                    
                    print("*" * 50)
                        
                
                print("Best Accuracy :", best_acc)
                model.load_state_dict(best_weights)
                model.eval()
                print("Validation and Training:", accuracy(model, valid_loader), accuracy(model, train_loader))
                test_loss = 0.0
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(test_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs, acts = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                
                    test_acc = accuracy(model, test_loader) 
                    layers = get_activations(model, train_loader)
                
                test_loss = test_loss / len(test_loader)
                
                valid_loss = sum(validation_loss)/len(validation_loss)
                train_loss = sum(training_loss)/len(training_loss)
                
                training_accuracy = sum(train_accuracy) / len(train_accuracy)
                validation_accuracy = sum(valid_accuracy) / len(valid_accuracy)
                
                
                percent = percentmask(mask)
                percentremoved[masktype].append(percent)
                train_accuracies[masktype].append(training_accuracy)
                valid_accuracies[masktype].append(validation_accuracy)
                test_accuracies[masktype].append(test_acc)
                train_losses[masktype].append(train_loss)
                valid_losses[masktype].append(valid_loss)
                test_losses[masktype].append(test_loss)
                early_stopping[masktype].append(epoch)
                
                if printvalue:
                    print('Percentage remaining', percent, end = ' ')
                    print('Layer nodes:', [torch.sum(mask[i]) for i in mask.keys()], end = ' ')
                    if printmask:
                        print('Mask:', mask)
                    print('Train Acc:', training_accuracy, end = ' ')
                    print('Val Acc:', validation_accuracy, end = ' ')
                    print('Test Acc:', test_acc)
                    print('Train Loss:', train_loss, end = ' ')
                    print('Val loss:', valid_loss, end = ' ')
                    print('Test Loss:', test_loss)
                    print('Early stopping iteration:', epoch)
                    

                # Remove nodes for next iteration based on metric
                maskflatten = flattenmask(mask)
                # print(mask)
                for each in mask:
                    mask[each] = mask[each].cpu().numpy()

                mask = getmask(layers, mask, maskflatten, mask_type = masktype, 
                               percentile = percentile, printactivation = printactivation)
                
                for each in mask:
                    mask[each] = torch.tensor(mask[each]).to(device)
                
                # print(mask)
        cache = (percentremoved, train_accuracies, valid_accuracies, test_accuracies, train_losses, valid_losses, test_losses, early_stopping, oraclecomparison)
        printgraph(cache, 'evaluate' + str(trial), numtrials = trial+1)

    cache = (percentremoved, train_accuracies, valid_accuracies, test_accuracies, train_losses, valid_losses, test_losses, early_stopping, oraclecomparison)
    return cache


criterion = nn.CrossEntropyLoss()
cache = generatemodel(criterion, patience = 5, num_epochs = 100, 
                      numtrials = 1, percentile = 0.2, printvalue = True, 
                      printmask = False, printactivation = False)

modelname = 'cifar10_alexNet_evaluate'
printgraph(cache, modelname, numtrials = 1, oracle = False)
savefile(cache, modelname)