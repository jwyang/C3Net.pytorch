from __future__ import print_function
import argparse
import torch
import pdb
import time
import yaml
import os
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from lib.utils.net_utils import weights_normal_init, adjust_learning_rate
from configs import cfg
from lib import *
from ptflops import get_model_complexity_info
from thop import profile

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def graph_node_loss(graphs, target):
    loss = 0
    for graph in graphs:
        node = graph[0]
        B, N = node.shape
        mean_node = torch.mm(node, node.transpose(0, 1).contiguous())
        loss += (mean_node.sum() - torch.diag(mean_node).sum()) / N / N
    return loss

def graph_edge_loss(graphs, target):
    loss = 0
    for graph in graphs:
        edge = graph[1]
        B, N, _ = edge.shape
        edge = F.relu(edge)
        loss += (edge.mean(0).sum() - torch.diag(edge.mean(0)).sum()) / N / N
    return loss

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batch_time = AverageMeter('Time', ':3.3f')
    data_time = AverageMeter('Data', ':3.3f')
    mem_cost = AverageMeter('Mem', ':3.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    layer1_bf = AverageMeter('Corr@bf', ':3.2f')
    layer1_af = AverageMeter('Corr@af', ':3.2f')
    layer2_bf = AverageMeter('Corr@bf', ':3.2f')
    layer2_af = AverageMeter('Corr@af', ':3.2f')
    layer3_bf = AverageMeter('Corr@bf', ':3.2f')
    layer3_af = AverageMeter('Corr@af', ':3.2f')

    progress = ProgressMeter(len(train_loader), mem_cost, batch_time, data_time, losses, top1,
                             top5, layer1_bf, layer1_af, layer2_bf, layer2_af, layer3_bf, layer3_af,
                             prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        mem = torch.cuda.max_memory_allocated()
        mem_cost.update(mem / 1024 / 1024 / 1024)

        if model.net.name == "invcnn":
            loss = 0
            for out in output:
                out = F.log_softmax(out, dim=1)
                loss += F.nll_loss(out, target)
            loss /= len(output)
        else:
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)

        acc1, acc5 = compute_accuracy(output[-1] if model.net.name == "invcnn" else output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        layer1_bf.update(model.net.layer1.cn.corr_bf.item())
        layer1_af.update(model.net.layer1.cn.corr_af.item())           
        layer2_bf.update(model.net.layer2.cn.corr_bf.item())
        layer2_af.update(model.net.layer2.cn.corr_af.item())
        layer3_bf.update(model.net.layer3.cn.corr_bf.item())
        layer3_af.update(model.net.layer3.cn.corr_af.item())
        # import pdb; pdb.set_trace()
        # node_loss = graph_node_loss(graphs, target)
        # edge_loss = graph_edge_loss(graphs, target)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log.print_interval == 0:
            progress.print(batch_idx)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tmem: {:.3f}m\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset), mem / 1024 / 1024,
            #     100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    batch_time = AverageMeter('Time', ':3.3f')
    losses = AverageMeter('Loss', ':.3f')
    mem_cost = AverageMeter('Mem', ':3.3f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(len(test_loader), mem_cost, batch_time, losses, top1, top5,
                             prefix='Test: ')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if model.net.name == "invcnn":
                output = torch.stack(output, 1).max(1)[0]
            mem = torch.cuda.max_memory_allocated()
            mem_cost.update(mem / 1024 / 1024 / 1024)

            # measure accuracy and record loss
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # compute val loss
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target).item() # sum up batch loss
            test_loss += loss
            losses.update(loss, data.size(0))

            # compute accuracy
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args.log.print_interval == 0:
                progress.print(batch_idx)
    test_loss /= batch_idx
    data_size = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, data_size,
        100. * float(correct) / data_size))
    accuracy = 100. * float(correct) / data_size
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='specify training dataset')
    parser.add_argument('--session', type=int, default='1',
                        help='training session to recoder multiple runs')
    parser.add_argument('--arch', type=str, default='resnet110',
                        help='specify network architecture')
    parser.add_argument('--bs', dest="batch_size", type=int, default=128,
                        help='training batch size')
    parser.add_argument('--gpu0-bs', dest="gpu0_bs", type=int, default=0,
                        help='training batch size on gpu0')
    parser.add_argument('--add-ccn', type=str, default='no',
                        help='add cross neruon communication')
    parser.add_argument('--mgpus', type=str, default="no",
                        help='multi-gpu training')
    parser.add_argument('--resume', dest="resume", type=int, default=0,
                        help='resume epoch')


    args = parser.parse_args()
    cfg.merge_from_file(osp.join("configs", args.dataset + ".yaml"))
    cfg.dataset = args.dataset
    cfg.arch = args.arch
    cfg.add_cross_neuron = True if args.add_ccn == "yes" else False
    use_cuda = True if torch.cuda.is_available() else False
    cfg.use_cuda = use_cuda
    cfg.training.batch_size = args.batch_size
    cfg.mGPUs = True if args.mgpus == "yes" else False

    torch.manual_seed(cfg.initialize.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = create_data_loader(cfg)
    model = CrossNeuronNet(cfg)
    print("parameter numer: %d" % (count_parameters(model)))
    with torch.cuda.device(0):
        if args.dataset == "cifar100":
            flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
            # flops, params = profile(model, input_size=(1, 3, 32, 32))
        elif args.dataset == "imagenet":
            flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
            # flops, params = profile(model, input_size=(1, 3, 224, 224))
        print('Flops: {}'.format(flops))
        print('Params: {}'.format(params))

    model = model.to(device)

    # optimizer_policy = model.get_optim_policies()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if cfg.mGPUs:
        if args.gpu0_bs > 0:
            model = BalancedDataParallel(args.gpu0_bs, model).to(device)
        else:
            model = nn.DataParallel(model).to(device)

    lr = cfg.optimizer.lr
    checkpoint_tag = osp.join("checkponts", args.dataset, args.arch)
    if not osp.exists(checkpoint_tag):
        os.makedirs(checkpoint_tag)

    if args.resume > 0:
        ckpt_path = osp.join(checkpoint_tag,
            ("ccn" if cfg.add_cross_neuron else "plain") + "_{}_{}.pth".format(args.session, args.resume))
        print("resume model from {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        print("resume model succesfully")
        acc = test(cfg, model, device, test_loader)

    best_acc = 0
    for epoch in range(args.resume + 1, cfg.optimizer.max_epoch + 1):
        if epoch in cfg.optimizer.lr_decay_schedule:
            adjust_learning_rate(optimizer, cfg.optimizer.lr_decay_gamma)
            lr *= cfg.optimizer.lr_decay_gamma
        print('Train Epoch: {} learning rate: {}'.format(epoch, lr))
        tic = time.time()
        train(cfg, model, device, train_loader, optimizer, epoch)
        acc = test(cfg, model, device, test_loader)
        time_cost = time.time() - tic
        if acc > best_acc:
            best_acc = acc
        print('\nModel: {} Best Accuracy-Baseline: {}\tTime Cost per Epoch: {}\n'.format(
            checkpoint_tag + ("ccn" if args.add_ccn == "yes" else "plain"),
            best_acc,
            time_cost))

        if epoch % cfg.log.checkpoint_interval == 0:
            checkpoint = {"arch": cfg.arch,
                          "model": model.state_dict(),
                          "epoch": epoch,
                          "lr": lr,
                          "test_acc": acc,
                          "best_acc": best_acc}
            torch.save(checkpoint, osp.join(checkpoint_tag,
                ("ccn" if cfg.add_cross_neuron else "plain") + "_{}_{}.pth".format(args.session, epoch)))


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
