from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim

import models
from torch.autograd import Variable

def save_state(model, best_acc):
    #print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        
        # process the weights including binarization
        bin_params.binarize()
        
        # forwarding
        if not args.cpu:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_params.restore()
        bin_params.updateBinaryGradWeight()
        
        optimizer.step()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / float(len(trainloader)), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    testing = True

    bin_params.binarize()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
                                    
        output = model(data, testing)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    bin_params.restore()
    acc = 100. * correct.item() / float(len(testloader.dataset))

    # Save the model params if the accuracy is the highest yet
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= float(len(testloader.dataset))
    print('\nTest Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(testloader.dataset), 100. * correct.item() / float(len(testloader.dataset)))
    )
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [40,80,120,160]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    return


#-----------------------------------------------------------------
if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default=os.getcwd()+'/datasets' ,#'./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    print(args.data+'/data_batch_1')

    # prepare the data
    if not os.path.isfile(args.data+'/data_batch_1'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    trainset = data.dataset(root=args.data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    testset = data.dataset(root=args.data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = models.NiN()
    elif args.arch == 'AlexNet':
        model = models.AlexNet()
    elif args.arch == 'VGGNet':
        model = models.VGGNet()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator for the model parameters
    bin_params = util.BinarizedParams(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # train
    for copies in range(10):
        for epoch in range(1, 201):
            adjust_learning_rate(optimizer, epoch)
            train(epoch)
            test()
