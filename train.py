import os
import argparse
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from timm.data import resolve_data_config
from conf import settings
from models.main import get_network
from utils.dataloaders import get_train_dataloader, get_test_dataloader
from utils.general import WarmUpLR, set_random_seed


def train(epoch):

    start = time.time()
    net.train()

    train_bar = enumerate(train_loader)
    if not args.batch_log:
        train_bar = tqdm(train_bar, total=len(train_loader), desc=f'Training Epoch[{epoch}/{args.epochs}]', unit='B')

    train_loss = 0.0
    for batch_index, (images, labels) in train_bar:

        if args.device:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_iter = (epoch - 1) * iter_per_epoch + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if args.batch_log:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(images),
                total_samples=train_num
            ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            # print(warmup_scheduler.get_lr())
            warmup_scheduler.step()

    finish = time.time()
    if args.batch_log:
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    else:
        print('\tLoss: {:0.4f}\tLR: {:0.6f}'.format(train_loss / train_num, optimizer.param_groups[0]['lr']))
        train_bar.close()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    val_bar = val_loader
    if not args.batch_log:
        val_bar = tqdm(val_loader, total=len(val_loader), desc=f'Test set')

    val_loss = 0.0     # cost function error
    correct = 0.0

    for (images, labels) in val_bar:

        if args.device:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        val_loss += loss.item()
        _, preds = outputs.max(1)
        print(outputs,111)
        print(preds,222)
        print(labels,333)
        correct += preds.eq(labels).sum()
    accuracy = correct / val_num
    val_loss = val_loss / val_num

    finish = time.time()
    # if args.device:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    if args.batch_log:
        print('Evaluating Network.....')
        print('Val set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            val_loss,
            accuracy,
            finish - start
        ))
        print()
    else:
        print('Val set:, Average loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, accuracy))
        val_bar.close()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Val/Average loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)

    return accuracy


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--package', action='store_true', default="CIFAR10", help='package dataset(CIFAR10/CIFAR100)')
    parser.add_argument('--data_dir', metavar='DIR', default=r"./data", help='path to dataset')

    # Model parameters
    parser.add_argument('--net', default='resnet18', type=str, help='net type')
    parser.add_argument('--num_classes', type=int, default=10, help='number for label classes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or None')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--resume', default='', type=str, metavar='filename in checkpoint(default: resnet18-last.pth)')

    # Learning rate schedule parameters
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warm', type=int, default=15, help='warm up training phase')

    # Optimizer parameters

    # Misc
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--seed', type=int, default=10, help='random seed (default: 10 or None)')
    parser.add_argument('--batch_log', action='store_true', default=False, help='print training batch log or progress bar')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [int(i) for i in args.device if i != ',']))
    set_random_seed(seed=args.seed, deterministic=False, benchmark=args.img_size)
    net = get_network(args)
    # ································

    # data preprocessing:
    data_config = resolve_data_config(vars(args))

    train_loader = get_train_dataloader(
        package=args.package,
        data_dir=args.data_dir,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = get_test_dataloader(
        package=args.package,
        data_dir=args.data_dir,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    train_num = len(train_loader.dataset)
    val_num = len(val_loader.dataset)
    print("using {} images for training, {} images for validation.\n".format(train_num, val_num))
    print("net: {}  epochs: {}  lr: {}  num_classes: {}  input_size:({}, {}, {}, {})".format(args.net, args.epochs, args.lr,
                                                                                             args.num_classes, args.batch_size,
                                                                                             3, args.img_size, args.img_size))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # learning rate decay
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.6)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # create folder to save model and tensorboard log
    save_path = os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # checkpoint_path = os.path.join(save_path, '{net}-{epoch}-{type}.pth')
    checkpoint_path = os.path.join(save_path, '{net}-{type}.pth')

    writer = SummaryWriter(log_dir=save_path)
    input_tensor = torch.Tensor(2, 3, args.img_size, args.img_size)
    if args.device:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    if args.resume:
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.resume)  # args.net
        assert os.path.exists(weights_path), "file {} does not exist.".format(weights_path)
        print('loading weights file {} to resume training.....'.format(weights_path))
        checkpoint_model = torch.load(weights_path)

        net.load_state_dict(checkpoint_model['model_state_dict'])
        optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
        # train_scheduler.load_state_dict(checkpoint_model['lr_scheduler_state_dict'])
        resume_epoch = checkpoint_model['epoch']
        print("====>loaded checkpoint (epoch{})".format(resume_epoch))


    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if not args.resume:
            if epoch > args.warm:
                train_scheduler.step(epoch)     # 学习率调整,在优化器更新后应用
        else:
            if epoch > resume_epoch + 1:
                train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        checkpoint_dict = {"model_state_dict": net.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(),
                           # "lr_scheduler_state_dict": train_scheduler.state_dict(),
                           "epoch": epoch}
        # save best performance model
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        if best_acc < acc:
            # weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            weights_path = checkpoint_path.format(net=args.net, type='best')
            # print('saving weights file to {}'.format(weights_path))
            torch.save(checkpoint_dict, weights_path)
            best_acc = acc

        # if not epoch % settings.SAVE_EPOCH:
        # weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
        weights_path = checkpoint_path.format(net=args.net, type='last')
        # print('saving weights file to {}'.format(weights_path))
        torch.save(checkpoint_dict, weights_path)

    writer.close()
