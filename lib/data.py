import torch
from torchvision import datasets, transforms
from .utils.imagenet import Loader
from .utils.cub2011 import Cub2011
def get_cifar10(opts):
    kwargs = {'num_workers': opts.training.workers, 'pin_memory': True} if opts.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                       ])),
        batch_size=opts.training.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                       ])),
        batch_size=opts.test.batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def get_cifar100(opts):
    kwargs = {'num_workers': opts.training.workers, 'pin_memory': True} if opts.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                       ])),
        batch_size=opts.training.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                       ])),
        batch_size=opts.test.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_cub2011(opts):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    kwargs = {'num_workers': opts.training.workers, 'pin_memory': True} if opts.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        Cub2011('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.Resize(256),
                           transforms.RandomCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize,
                       ])),
        batch_size=opts.training.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Cub2011('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
                       ])),
        batch_size=opts.test.batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def get_imagenet(opts):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     opts.data.traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # if opts.training.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=opts.training.batch_size, shuffle=(train_sampler is None),
    #     num_workers=opts.training.workers, pin_memory=True, sampler=train_sampler)

    # import pdb; pdb.set_trace()

    train_loader = Loader('train', batch_size=opts.training.batch_size, num_workers=opts.training.workers)

    val_dataset = datasets.ImageFolder(
        opts.data.valdir,
        transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=opts.test.workers, pin_memory=True)
    # val_loader = Loader('val', batch_size=opts.test.batch_size, num_workers=opts.test.workers)
    return train_loader, val_loader

def create_data_loader(opts):
    if opts.dataset == "cifar10":
        return get_cifar10(opts)
    elif opts.dataset == "cifar100":
        return get_cifar100(opts)
    elif opts.dataset == "cub2011":
        return get_cub2011(opts)
    elif opts.dataset == "imagenet":
        return get_imagenet(opts)
    else:
        raise ValueError("Unknow dataset")
