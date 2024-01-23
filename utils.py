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
import matplotlib.pyplot as plt

import copy

def printgraph(cache, name, numtrials = 15, oracle = False):
    """Function to print graph
    Input: 
    cache - Cache containing accuracies, losses, early stopping iteration and oracle comparison
    name - name of model which is run
    numtrials - number of experiments conducted
    oracle - Boolean. Whether the oraclecomparison graph is required
    
    Outputs:
    Graphs for training, validation, test accuracy, early stopping iteration, oracle comparison (optional)"""
    
    # unpack caches
    percentremoved, train_accuracies, valid_accuracies, test_accuracies, train_losses, valid_losses, test_losses, early_stopping, oraclecomparison = cache
    # sort masktypes by alphabetical order
    masktypes = sorted(percentremoved.keys())

    # Set colors
    colors = {}
    colors['min'] = 'b'
    colors['max'] = 'r'
    colors['random'] ='g'
    colors['random_layer'] = 'y'
    colors['min_layer'] = 'c'
    colors['max_layer'] = 'm'
    colors['randominit'] = 'y'
    colors['oracle'] = 'm'
    
    # colors for percentage mask
    colors['0.1'] = 'b'
    colors['0.2'] = 'r'
    colors['0.3'] = 'g'
    colors['0.4'] = 'y'
    colors['0.5'] = 'c'
    colors['0.9'] = 'm'
    
    # format for various masks
    fmt = {}
    fmt['min'] = '-'
    fmt['max'] = '-'
    fmt['random'] ='--'
    fmt['random_layer'] = '--'
    fmt['min_layer'] = '-'
    fmt['max_layer'] = '-'
    fmt['randominit'] = '--'
    fmt['minfast'] = '-'
    fmt['oracle'] = '-'
    
    fmt['0.1'] = '-'
    fmt['0.2'] = '-'
    fmt['0.3'] = '-'
    fmt['0.4'] = '-'
    fmt['0.5'] = '-'
    fmt['0.9'] = '-'
    
    rc('font', family = 'STIXGeneral')
    rc('xtick', labelsize=10) 
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 14})
    
    # Plot figures for training accuracy
    plt.figure()
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.1','0.2','0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], rotation = 45)
    for masktype in masktypes:     
        length = len(percentremoved[masktype])//numtrials
        mean = []
        std = []
        for i in range(length):
            mean.append(np.mean(train_accuracies[masktype][i::length]))
            std.append(1.96*np.std(train_accuracies[masktype][i::length])/np.sqrt(numtrials))
        pr = [tensor.item() for tensor in percentremoved[masktype][:length]]
        plt.errorbar(pr, mean, yerr = std, fmt = fmt[masktype], capsize = 2, alpha = 0.5, color=colors[masktype], label = masktype)        
    plt.ylabel('Training accuracy')
    plt.xlabel('Proportion of nodes/filters remaining')
    plt.legend(loc = 'lower left')
    plt.savefig('/scratch/smanduru/ML_688/DropNet/Images/training_accuracy_{}.png'.format(name))

    # Plot figures for validation accuracy
    plt.figure()
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.1','0.2','0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], rotation = 45)
    for masktype in masktypes:
        length = len(percentremoved[masktype])//numtrials
        mean = []
        std = []
        for i in range(length):
            mean.append(np.mean(valid_accuracies[masktype][i::length]))
            std.append(1.96*np.std(valid_accuracies[masktype][i::length])/np.sqrt(numtrials))
        pr = [tensor.item() for tensor in percentremoved[masktype][:length]]
        plt.errorbar(pr, mean, yerr = std, fmt = fmt[masktype], capsize = 2, alpha = 0.5, color=colors[masktype], label = masktype)
    plt.ylabel('Validation accuracy')
    plt.xlabel('Proportion of nodes/filters remaining')
    plt.legend(loc = 'lower left')
    plt.savefig('/scratch/smanduru/ML_688/DropNet/Images/validation_accuracy_{}.png'.format(name))

    # Plot figures for test accuracy
    plt.figure()
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.1','0.2','0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], rotation = 45)
    for masktype in masktypes:
        length = len(percentremoved[masktype])//numtrials
        mean = []
        std = []
        for i in range(length):
            mean.append(np.mean(test_accuracies[masktype][i::length]))
            std.append(1.96*np.std(test_accuracies[masktype][i::length])/np.sqrt(numtrials))
        pr = [tensor.item() for tensor in percentremoved[masktype][:length]]
        plt.errorbar(pr, mean, yerr = std, fmt = fmt[masktype], capsize = 2, alpha = 0.5, color=colors[masktype], label = masktype)
    plt.ylabel('Test accuracy')
    plt.xlabel('Proportion of nodes/filters remaining')
    plt.legend(loc = 'lower left')
    plt.savefig('/scratch/smanduru/ML_688/DropNet/Images/test_accuracy_{}.png'.format(name))

    if oracle:
        # Plot figures for oracle comparison
        plt.figure()
        plt.gca().invert_xaxis()
        plt.xscale('log')
        plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.1','0.2','0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], rotation = 45)
        for masktype in masktypes:
            length = len(percentremoved[masktype])//numtrials
            mean = []
            std = []
            for i in range(length):
                mean.append(np.mean(oraclecomparison[masktype][i::length]))
                std.append(1.96*np.std(oraclecomparison[masktype][i::length])/np.sqrt(numtrials))
            pr = [tensor.item() for tensor in percentremoved[masktype][:length]]
            plt.errorbar(pr, mean, yerr = std, fmt = fmt[masktype], capsize = 2, alpha = 0.5, color=colors[masktype], label = masktype)
        plt.ylabel('Test accuracy')
        plt.xlabel('Proportion of nodes/filters remaining')
        plt.legend(loc = 'lower left')
        plt.savefig('/scratch/smanduru/ML_688/DropNet/Images/oracle_comparison_{}.png'.format(name))

    # Plot figures for early stopping iteration
    plt.figure()
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.1','0.2','0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], rotation = 45)
    for masktype in masktypes:
        length = len(percentremoved[masktype])//numtrials
        mean = []
        std = []
        for i in range(length):
            mean.append(np.mean(early_stopping[masktype][i::length]))
            std.append(1.96*np.std(early_stopping[masktype][i::length])/np.sqrt(numtrials))
        pr = [tensor.item() for tensor in percentremoved[masktype][:length]]
        plt.errorbar(pr, mean, yerr = std, fmt = fmt[masktype], capsize = 2, alpha = 0.5, color=colors[masktype], label = masktype)
        plt.ylabel('Early stopping iteration')
        plt.xlabel('Proportion of nodes/filters remaining')
        plt.legend(loc = 'lower left')
        plt.savefig('/scratch/smanduru/ML_688/DropNet/Images/early_stopping_{}.png'.format(name))


def percentmask(mask):
    """Returns the percentage of mask that contains 1s"""
    nummask = 0
    totalmask = 0

    for v in mask.values():
        nummask += torch.sum(v)
        totalmask += v.numel()

    return nummask / totalmask

def resetmask(mask):
    """Resets mask to initial start state of all ones"""
    for k, v in mask.items():
        mask[k] = torch.ones_like(v)

    return mask

def flattenmask(mask):
    """Returns the nodes of mask flattened
    Input:
    mask - the mask position, in dictionary form
    Output:
    maskflatten - the mask position, flattened out in 1D array
    """
    maskflatten = []
    for k, v in mask.items():
        curmask = v.flatten().cpu().detach().numpy() # Convert to NumPy array and detach from GPU if applicable
        maskflatten = np.hstack([maskflatten, curmask])

    return maskflatten

def getmask(layers, mask, maskflatten, mask_type = 'min', percentile = 0.2, printactivation = False, dropOne = False):
    """ Updates mask after each training cycle
    Inputs:
    layers - Predicted node value per layer
    mask - current mask
    mask_type - type of mask: min, max, random, min_layer, max_layer, random_layer
    maskflatten - Flattened masked indices per layer (1 for mask, 0 for no mask)
    percentile - percentage of nodes remaining to be masked
    printactivation - Boolean. Whether to print the activations per layer
    dropOne - Boolean. Whether to drop only one node/filter at a time
    
    Output:
    mask - final masks after masking percentile proportion of remaining nodes
    """
    nodevalues = []
    layermeans = {}
    
    # if only drop one, then percentile is 0
    if dropOne:
        percentile = 0
    
    print("mask length", len(mask))
    # if only one layer
    if(len(mask)==1):
        layermeans[0] = np.mean(np.abs(layers), axis = 0).ravel()
        nodevalues = np.hstack([nodevalues, layermeans[0]])
        if printactivation:
            print('Layer activations:', layermeans[0])
        
    # if more than one layer
    else:
        for i in range(len(mask)):
            layermeans[i] = np.mean(np.abs(layers[i]), axis = 0).ravel()
            nodevalues = np.hstack([nodevalues, layermeans[i]])
            if printactivation:
                print('Layer activations:', layermeans[i])

    # remove only those in maskindex
    maskflatten = np.ravel(np.where(maskflatten == 1))
    
    # find out the threshold node/filter value to remove
    if len(maskflatten) > 0:
        # for max mask
        if mask_type == 'max':
            sortedvalues = -np.sort(-nodevalues[maskflatten])
            index = int((percentile)*len(sortedvalues))
            maxindex = sortedvalues[index]

        # for min or % mask
        else:
            sortedvalues = np.sort(nodevalues[maskflatten])
            index = int(percentile*len(sortedvalues))
            maxindex = sortedvalues[index]
                           
    # Calculate the number of nodes to remove
    nummask = 0
    
    for v in mask.values():
        nummask += np.sum(v)
    
    totalnodes = int((percentile)*nummask)
    
    if dropOne:
        totalnodes = 1

    # remove at least one node
    if (totalnodes == 0):
        totalnodes = 1
    
    # identify the indices to drop for random mask
    if mask_type == 'random':
        indices = np.random.permutation(maskflatten)
        # take only the first totalnodes number of nodes
        indices = indices[:totalnodes]
        
        dropmaskindex = {}
        startindex = 0
        # assign nodes/filters to drop for each layer in dropmaskindex
        for k, v in mask.items():
            nummask += np.sum(v)
            dropmaskindex[k] = indices[(indices>=startindex) & (indices < startindex + len(v))] - startindex
            startindex += len(v)
        
    for i, layermean in layermeans.items():

        #only if there is something to drop in current mask
        if(np.sum(mask[i])>0):
            # Have different indices for different masks
            if mask_type == 'max':
              indices = np.ravel(np.where(layermean>=maxindex))
              curindices = np.ravel(np.where(mask[i].ravel()))
              indices = [j for j in indices if j in curindices]
            # global random mask or layer random mask
            elif mask_type == 'random_layer':
              indices = np.ravel(np.where(mask[i].ravel()))
              curindices = np.ravel(np.where(mask[i].ravel()))
            elif mask_type == 'random':
              indices = dropmaskindex[i]
              curindices = np.ravel(np.where(mask[i].ravel()))
            # layer-wise max mask
            elif mask_type == 'max_layer':
              sortedvalues = -np.sort(-layermean[mask[i]==1])
              index = int((percentile)*len(sortedvalues))
              maxindex = sortedvalues[index]
              indices = np.ravel(np.where(layermean>=maxindex))
              curindices = np.ravel(np.where(mask[i].ravel()))
              indices = [j for j in indices if j in curindices]
            # layer-wise min mask
            elif mask_type == 'min_layer':
              sortedvalues = np.sort(layermean[mask[i]==1])
              index = int((percentile)*len(sortedvalues))
              maxindex = sortedvalues[index]
              indices = np.ravel(np.where(layermean<=maxindex))
              curindices = np.ravel(np.where(mask[i].ravel()))
              indices = [j for j in indices if j in curindices]
            # if this is min mask or % based mask
            else:
              indices = np.ravel(np.where(layermean<=maxindex))
              curindices = np.ravel(np.where(mask[i].ravel()))
              indices = [j for j in indices if j in curindices]
                
        else:
          #default
          indices = np.ravel(np.where(mask[i]==1))

        # shuffle the indices only if we are not dropping one node/filter
        if (dropOne == False):
          indices = np.random.permutation(indices)

        newmask = mask[i].ravel()

        # for layer masks, total nodes dropped is by percentile of the layer of each mask
        if(mask_type == 'random_layer') or mask_type == 'min_layer' or mask_type == 'max_layer':
            initialpercent = np.sum(mask[i])*1.0/len(mask[i].ravel())
            totalnodes = int(initialpercent*(percentile)*len(mask[i].ravel()))

            # remove at least 1 node
            if (totalnodes == 0):
                totalnodes = 1

        if(len(indices)>0):

            # remove at most totalnodes number of nodes
            if(len(indices)>totalnodes):
                indices = indices[:totalnodes]

            # remove nodes
            newmask[indices] = 0

            # updated totalnodes to be removed
            totalnodes = totalnodes - len(indices)

        # reshape to fit new mask
        mask[i] = newmask.reshape(mask[i].shape)

    return mask

def comparemask(mask1, mask2):
    """ Compares how similar both masks (mask1, mask2) are and returns a percentage similarity """
    count = 0
    totalcount = 0
    for k, v in mask1.items():
        count += torch.sum(mask1[k] == mask2[k])
        totalcount += torch.numel(mask1[k])
        
    return float(count)/totalcount


def savefile(cache, name):
    """ Function which saves the cache """
    with open('/scratch/smanduru/ML_688/DropNet/Caches/'+name+'.p','wb') as outfile:
        pickle.dump(cache, outfile)
        
def loadfile(name):
    """ Function which loads the cache """
    with open('/scratch/smanduru/ML_688/DropNet/Caches/'+name+'.p','rb') as infile:
        cache = pickle.load(infile)
    return cache