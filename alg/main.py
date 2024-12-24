import argparse
import logging
import time
import os

import sys
sys.path.append("../")
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from fedlab.utils.functional import setup_seed, AverageMeter
from fedlab.core.standalone import StandalonePipeline
from fedlab.models.resnet import *
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedcfa import FedCFServerHandler, FedCFSerialClientTrainer


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser('FedCFA', add_help=False)

    parser.add_argument('--setup_seed', default=1234, type=int, help='setup_seed')
    parser.add_argument('--logpath', default='../log/', type=str, help='logfile')
    
    # fedcfa
    parser.add_argument('--topk', default=24, type=int, help='topk')
    parser.add_argument('--fedcfa_rate', default='1:5:5', type=str, help='fedcfa rate')
    parser.add_argument('--lc', default=0.1, type=float, help='l_corr')
    
    # train
    parser.add_argument('--model', default='ResNetHook2', type=str, help='model')
    parser.add_argument('--alg', default='fedcfa', type=str, help='alg')
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float, help='lr')
    
    # server
    parser.add_argument('--com_round', default=500, type=int, help='com_round')
    parser.add_argument('--sample_ratio', default=0.2, type=float, help='sample_ratio')
    parser.add_argument('--target_acc', default=1.0, type=float, help='target_acc')
    
    # dataset
    parser.add_argument('--total_client', default=60, type=int, help='total_client')
    parser.add_argument('--alpha', default=0.6, type=float, help='alpha')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--preprocess', type=str2bool, default=True)
    # parser.add_argument('--data_root', default='../datasets/CIFAR10/', type=str, help='data_root')
    # parser.add_argument('--data_path', default='../datasets/CIFAR10/fed60_06/', type=str, help='data_path')
    parser.add_argument('--data_name', default='CIFAR10', type=str, help='data_name')
    parser.add_argument('--partition', default='dirichlet', type=str, help='partition')
    parser.add_argument('--balance', type=str2bool, default=True, help='balance')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle')

    args=parser.parse_args()
    return args

def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs,_ = outputs
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
    
    def main(self, logger, com_round, target_acc=1.0):
        acc_max=0
        p_round = 0
        best_round = 0
        losses = []
        acces = []
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            if acc > acc_max:
                acc_max = acc
                best_round = p_round

            if p_round<10 or p_round % 10 == 0 or p_round==com_round-1:
                logger.info(f"round {p_round} - loss {loss:.4f}, test acc {acc:.4f}, MaxAcc {best_round}-{acc_max:.4f}")

            acces.append(acc)
            losses.append(loss)

            if acc >= target_acc:
                logger.info(f"target_round {p_round}, test_acc {acc:.4f}, target_acc {target_acc:.4f}")
                break
            
            p_round += 1
        # print("best test accuracy {:.4f}".format(acc_max))
        logger.info(f"best round - best test accuracy {acc:.4f}({best_round},{acc_max:.4f})")
        return losses, acces

def main(args):
    setup_seed(args.setup_seed)
    torch.set_num_threads(12)

    logfile = args.logpath + args.alg + '.log'
    os.makedirs(args.logpath, exist_ok=True)
    logger = logging.getLogger('fedlab')
    logger.setLevel(logging.NOTSET)
    fh = logging.FileHandler(logfile, mode='a', encoding=None,delay=False)
    fh.setLevel(logging.NOTSET)
    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    args.data_path = f"../datasets/{args.data_name}/fed{args.total_client}_{args.partition if args.partition=='iid' else str(args.alpha).replace('.', '')}/"
    if 'fedcfa' in args.alg:
        args.model = 'ResNetHook2'
    else:
        args.model = 'ResNet'
    logger.info('----------------------------')
    logger.info('setup_seed: '+str(args.setup_seed))
    logger.info('logfile: '+logfile)
    logger.info('topk: '+str(args.topk))
    logger.info('fedcfa_rate: '+str(args.fedcfa_rate))
    logger.info('lc: '+str(args.lc))
    logger.info('model: '+str(args.model))
    logger.info('alg: '+str(args.alg))
    logger.info('cuda: '+str(args.cuda))
    logger.info('epochs: '+str(args.epochs))
    logger.info('batch_size: '+str(args.batch_size))
    logger.info('lr: '+str(args.lr))
    logger.info('com_round: '+str(args.com_round))
    logger.info('total_client: '+str(args.total_client))
    logger.info('alpha: '+str(args.alpha))
    logger.info('seed: '+str(args.seed))
    logger.info('preprocess: '+str(args.preprocess))
    logger.info('data_name: '+str(args.data_name))
    logger.info('data_path: '+str(args.data_path))
    logger.info('shuffle: '+str(args.shuffle))
    logger.info('partition: '+str(args.partition))
    logger.info('target_acc: '+str(args.target_acc))
    
    if ('Hook' in args.model and not 'fedcfa' in args.alg) or (not 'Hook' in args.model and 'fedcfa' in args.alg):
        print('Model and algorithm do not match')
        return

    if args.data_name == 'CIFAR10':
        num_classes = 10
    elif args.data_name == 'CIFAR100':
        num_classes = 100
    if 'ResNetHook' in args.model:
        print('ResNetHook', args.model[-1])
        model = ResNetHookn(ResBlock, num_classes=num_classes, n=int(args.model[-1]))
    elif args.model == 'ResNet':
        model = ResNet(ResBlock, num_classes=num_classes)
    else:
        print(f'{args.model} does not exist')
        return

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if args.data_name == 'CIFAR10':
        if args.partition == 'dirichlet':
            fed_data = PartitionedCIFAR10(root="../datasets/CIFAR10/",
                            path=args.data_path,
                            dataname="CIFAR10",
                            num_clients=args.total_client,
                            balance=None,
                            partition="dirichlet",
                            dir_alpha=args.alpha,
                            seed=args.seed,
                            preprocess=args.preprocess,
                            download=True,
                            verbose=True,
                            shuffle=args.shuffle,
                            transform=transform_cifar)
        elif args.partition == 'iid':
            fed_data = PartitionedCIFAR10(root="../datasets/CIFAR10/",
                            path=args.data_path,
                            dataname="CIFAR10",
                            num_clients=args.total_client,
                            balance=args.balance,
                            partition="iid",
                            seed=args.seed,
                            preprocess=args.preprocess,
                            download=True,
                            verbose=True,
                            shuffle=args.shuffle,
                            transform=transform_cifar)
        else:
            print(f'The partition {args.partition} does not exist')
            return
    elif args.data_name == 'CIFAR100':
        if args.partition == 'dirichlet':
            fed_data = PartitionCIFAR(root="../datasets/CIFAR100/",
                            path=args.data_path,
                            dataname="cifar100",
                            num_clients=args.total_client,
                            balance=None,
                            partition="dirichlet",
                            dir_alpha=args.alpha,
                            seed=args.seed,
                            preprocess=args.preprocess,
                            download=True,
                            verbose=True,
                            shuffle=args.shuffle,
                            transform=transform_cifar)
        elif args.partition == 'iid':
            fed_data = PartitionCIFAR(root="../datasets/CIFAR100/",
                            path=args.data_path,
                            dataname="cifar100",
                            num_clients=args.total_client,
                            balance=args.balance,
                            partition="iid",
                            seed=args.seed,
                            preprocess=args.preprocess,
                            download=True,
                            verbose=True,
                            shuffle=args.shuffle,
                            transform=transform_cifar)
        else:
            print(f'The partition {args.partition} does not exist')
            return

    if args.alg == 'fedcfa':
        if not args.model[:-1] == 'ResNetHook':
            print('fedcfa-ResNet should be ResNetHook')
            return
        if '.' in args.fedcfa_rate:
            fedcfa_rate = [float(a) for a in args.fedcfa_rate.split(':')]
        else:
            fedcfa_rate = [int(a) for a in args.fedcfa_rate.split(':')]
        print('fedcfa_rate',fedcfa_rate)
        handler = FedCFServerHandler(model=model,
                                     global_round=args.com_round, 
                                     sample_ratio=args.sample_ratio, cuda=args.cuda)
        trainer = FedCFSerialClientTrainer(model, args.total_client, cuda=args.cuda, topk=args.topk, num_classes=num_classes, fedcfa_rate=fedcfa_rate)
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    elif args.alg == 'fedavg':
        handler = SyncServerHandler(model=model, 
                                    global_round=args.com_round, 
                                    sample_ratio=args.sample_ratio, cuda=args.cuda)
        trainer = SGDSerialClientTrainer(model, args.total_client, cuda=args.cuda)
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    else:
        print('The algrithm does not exist')
        return
    
    trainer.setup_dataset(fed_data)
    if args.data_name == 'CIFAR10':
        test_data = torchvision.datasets.CIFAR10(root="../../FedCFA/datasets/CIFAR10/",
                                                train=False,
                                                transform=transform_cifar)
    elif args.data_name == 'CIFAR100':
        test_data = torchvision.datasets.CIFAR100(root="../../FedCFA/datasets/CIFAR100/",
                                                train=False,
                                                transform=transform_cifar)
    test_loader = DataLoader(test_data, batch_size=1024)
    standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
    start = time.time()

    losses, acces = standalone_eval.main(logger, args.com_round, args.target_acc)
   
    run_time = time.time()-start
    logger.info(f'花费时间 {run_time//3600}h {run_time%3600//60}m {run_time%60}s')

if __name__=='__main__':
    args = get_args()
    main(args)

    print('end')