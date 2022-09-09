from GLFC import GLFC_model
from ResNet import resnet18_cbam
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from ISCXVPN2016 import *
from option import args_parser
from torchviz import make_dot
from module_ISCX import *
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset

args = args_parser()

feature_extractor = ISCX_module()
model_g = network(args.numclass, feature_extractor)
model_g = model_to_device(model_g, False, args.device)

train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# train_dataset = ISCXVPN2016('./glfc_img_train', transform=train_transform)
# test_dataset = ISCXVPN2016('./glfc_img_test', test_transform=test_transform, train=False)

train_dataset = dset.ImageFolder("./glfc_img_train", transform=train_transform)
test_dataset = dset.ImageFolder("./glfc_img_test", transform=train_transform)





def model_to_device(model, parallel, device):
    if parallel:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        card = torch.device("cuda:{}".format(device))
        model.to(card)
    return model

# def efficient_old_class_weight_ISCX(output, label, device):
#     pred = torch.sigmoid(output)
    
#     N, C = pred.size(0), pred.size(1)

#     class_mask = pred.data.new(N, C).fill_(0)
#     class_mask = Variable(class_mask)
#     ids = label.view(-1, 1)
#     class_mask.scatter_(1, ids.data, 1.)

#     target = get_one_hot(label, 15, device)
#     g = torch.abs(pred.detach() - target)
#     g = (g * class_mask).sum(1).view(-1, 1)

#     if len(self.learned_classes) != 0:
#         for i in self.learned_classes:
#             ids = torch.where(ids != i, ids, ids.clone().fill_(-1))

#         index1 = torch.eq(ids, -1).float()
#         index2 = torch.ne(ids, -1).float()
#         if index1.sum() != 0:
#             w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
#         else:
#             w1 = g.clone().fill_(0.)
#         if index2.sum() != 0:
#             w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
#         else:
#             w2 = g.clone().fill_(0.)

#         w = w1 + w2
    
#     else:
#         w = g.clone().fill_(1.)

#     return w

def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def _compute_loss_ISCX(model, imgs, label, device):
    output = model(imgs)
    target = get_one_hot(label, 15, device)
    output, target = output.cuda(device), target.cuda(device)
    # print("target: ", target)
    # print("output: ", output)

    loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
    return loss_cur

# train model
def model_global_train_ISCX(model_g, train_dataset, device):
    model_g = model_to_device(model_g, False, device)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128)

    opt = optim.SGD(model_g.parameters(), lr=0.05, weight_decay=0.00001)
    
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(device), target.cuda(device)
        loss_value = _compute_loss_ISCX(model_g, images, target, device)
        opt.zero_grad()
        loss_value.backward()
        opt.step()


def model_global_eval_ISCX(model_g, test_dataset, device):
    model_g = model_to_device(model_g, False, device)
    model_g.eval()
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model_g.train()
    return accuracy

    
for ep in range(args.epochs_global):
    if ep % 10 == 0:
        print("*" * 60)
        acc = model_global_eval_ISCX(model_g, test_dataset, args.device)
        print('classification accuracy of model at round %d: %.3f \n' % (ep, acc))

    model_global_train_ISCX(model_g, train_dataset, args.device)


