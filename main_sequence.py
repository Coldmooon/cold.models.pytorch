import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import STMulti as models
from svhn import SVHN

class Evaluator(object):
    def __init__(self, val_loader):
        self._loader = val_loader

    def evaluate(self, model):
        model.eval()
        num_correct = 0
        needs_include_length = False

        for i, (images, length_labels, digits_labels) in enumerate(self._loader):
            images, length_labels, labels = (images.cuda(),
                                                    length_labels.cuda(),
                                                    [label.cuda() for label in digits_labels])


            length_logits, digits_logits = model(images)
            length_predictions = length_logits.data.max(1)[1]
            digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

            if needs_include_length:
                num_correct += (length_predictions.eq(length_labels.data) &
                                digits_predictions[0].eq(labels[0].data) &
                                digits_predictions[1].eq(labels[1].data) &
                                digits_predictions[2].eq(labels[2].data) &
                                digits_predictions[3].eq(labels[3].data) &
                                digits_predictions[4].eq(labels[4].data)).cpu().sum()
            else:
                num_correct += torch.sum(digits_predictions[0].eq(labels[0].data) &
                                digits_predictions[1].eq(labels[1].data) &
                                digits_predictions[2].eq(labels[2].data) &
                                digits_predictions[3].eq(labels[3].data) &
                                digits_predictions[4].eq(labels[4].data))

        accuracy = float(num_correct) / float(len(self._loader.dataset))

        return accuracy


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU index')
parser.add_argument('--se-reduce', default='16', type=int,
                    help='SE-Net reduce param')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='[imagenet|cifar10|cifar100]')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    devices = [int(s) for s in args.gpu.split(',') if s.isdigit()]
    nGPU = len(devices)
    devices = list(range(nGPU))

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, r=args.se_reduce)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](r=args.se_reduce)
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            if (nGPU == 1):
                model.cuda()
            else:
                model = torch.nn.DataParallel(model, device_ids=devices).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    print(args)
    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    lr_policy = list()
    params_collect = dict(model.named_parameters())
    for k, v in params_collect.items():
        if 'st' in k:
            lr_policy.append({'params': v, 'lr': 0.001, 'weight_decay': args.weight_decay})
        else:
            lr_policy.append({'params': v})
    
    optimizer = torch.optim.SGD(lr_policy,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if (args.dataset == 'imagenet'):
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif (args.dataset == 'cifar10'):
        to_normalized_tensor = [transforms.ToTensor(),
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))]
        data_augmentation = [transforms.RandomCrop(28, padding=0),
                             transforms.RandomHorizontalFlip()]

        transform = transforms.Compose(data_augmentation + to_normalized_tensor)

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    elif (args.dataset == 'svhn'):
        path_to_train_lmdb_dir = os.path.join(args.data, 'train.lmdb')
        path_to_val_lmdb_dir = os.path.join(args.data, 'val.lmdb')
        train_loader = torch.utils.data.DataLoader(SVHN(path_to_train_lmdb_dir),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(SVHN(path_to_val_lmdb_dir), batch_size=128, shuffle=False)

        # to_normalized_tensor = [transforms.ToTensor(),
        #                         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))]
        # data_augmentation = [transforms.RandomCrop(28, padding=0),
        #                      transforms.RandomHorizontalFlip()]
        #
        # transform = transforms.Compose(data_augmentation + to_normalized_tensor)
        #
        # trainset = datasets.CIFAR10(root='./data', train=True,
        #                                         download=True, transform=transform)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        #                                           shuffle=True, num_workers=2)
        #
        # testset = datasets.CIFAR10(root='./data', train=False,
        #                                        download=True, transform=transform)
        # val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
        #                                      shuffle=False, num_workers=2)
    else:
        print('No dataset named a ', args.dataset)
        exit(0)



    if args.evaluate:
        validate(val_loader, model, criterion)
        print(acc)
        return

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        print("Learning Rate: ", optimizer.param_groups[0]['lr'])
        # train for one epoch

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        print("\nBest Model: ", best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, length_labels, digits_labels) in enumerate(train_loader):

        input, length_labels, digits_labels = (input, length_labels.cuda(),[(digit_labels.cuda()) for digit_labels in digits_labels])

        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)

        # compute output
        # if not (input.is_cuda):
        #     input = input.cuda()
        if (len(args.gpu) == 1):
            length_logits, digits_logits = model(input.cuda())
        else:
            length_logits, digits_logits = model(input)

        loss = sequnece_loss(length_logits, digits_logits, length_labels, digits_labels)
        # loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = 0; # accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(0, input.size(0))
        top5.update(0, input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    acc = Evaluator(val_loader).evaluate(model)
    # switch to evaluate mode
    # model.eval()
    #
    # with torch.no_grad():
    #     end = time.time()
    #     for i, (input, target) in enumerate(val_loader):
    #         target = target.cuda(non_blocking=True)
    #
    #         # compute output
    #         if (len(args.gpu) == 1):
    #             output = model(input.cuda())
    #         else:
    #             output = model(input)
    #         loss = criterion(output, target)
    #
    #         # measure accuracy and record loss
    #         prec1, prec5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), input.size(0))
    #         top1.update(prec1[0], input.size(0))
    #         top5.update(prec5[0], input.size(0))
    #
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #
    #         if i % args.print_freq == 0:
    #             print('Test: [{0}/{1}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                    i, len(val_loader), batch_time=batch_time, loss=losses,
    #                    top1=top1, top5=top5))
    #
    #     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))

    return acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decays = 0
    decay = 0
    if (args.dataset == 'imagenet'):
        decays = epoch // 30
        decay = epoch % 30 == 0 and 1 or 0
    elif (args.dataset == 'cifar10' or args.dataset == 'cifar10'):
        decays = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
        decay = epoch == 122 and 1 or epoch == 81 and 1 or 0
    elif (args.dataset == 'svhn'):
        decay = epoch % 48 == 0 and 1 or 0
    else:
        print("No dataset named ", args.dataset)
        exit(-1)

    lr =  args.lr * pow(0.1, decays)

    for param_group in optimizer.param_groups:
        # param_group['lr'] = lr # global learning rate
        param_group['lr'] = param_group['lr'] * pow(0.1, decay) # learning rate for specific layer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def sequnece_loss(length_logits, digits_logits, length_labels, digits_labels):
    length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[0], digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[1], digits_labels[1])
    digit3_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[2], digits_labels[2])
    digit4_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[3], digits_labels[3])
    digit5_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[4], digits_labels[4])
    loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    return loss


if __name__ == '__main__':
    main()
