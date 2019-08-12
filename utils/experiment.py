from tensorboardX import SummaryWriter
from torch import nn, optim
from utils.functions import AverageMeter, ProgressMeter, Logger, accuracy
from constants import *
from datasets.dataset_c3d import DatasetC3D
from utils import video_transforms
from torch.utils.data import DataLoader
from models.C3D import C3D

import time
import os
import torch

EXPERIMENT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
CHECKPOINT_FORMAT = 'checkpoint_epoch{}.pth.tar'


class Experiment:

    experiment_date = None
    experiment_id = None
    experiment_path = None
    checkpoints_save_path = None
    summary_writer = None
    train_epoch_logger = None
    train_batch_logger = None
    validation_logger = None

    dataloader_train = None
    dataloader_test = None
    model = None
    optimizer = None
    criterion = None
    scheduler = None

    def __init__(self, opts, device):
        self.opts = opts
        self.device = device

        self.initialize_dataloaders()
        self.initialize_model()
        self.initialize_training()

        self.create_experiment_folders()
        self.initialize_loggers()

    def create_experiment_folders(self):
        # FIXME will be changed (related to multiple models)
        self.experiment_date = time.strftime("%d.%m.%Y-%H.%M.%S")
        self.experiment_id = 'experiment_' \
                             + self.experiment_date + '_' \
                             + self.dataloader_train.dataset.dataset + '_' \
                             + self.opts.models[0] + '_' \
                             + self.opts.modality + '_' \
                             + self.opts.sampling + '_' \
                             + 'clip' + str(self.opts.clip_length) + '_' \
                             + 'batch' + str(self.opts.batch_size) + '_' \
                             + 'optim' + self.opts.optimizer + '_' \
                             + 'loss' + self.opts.criterion + '_' \
                             + 'lr' + str(self.opts.learning_rate)

        self.experiment_path = os.path.join(EXPERIMENT_ROOT, self.experiment_id)
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        else:
            raise FileExistsError

        self.checkpoints_save_path = os.path.join(self.experiment_path, 'checkpoints')
        os.makedirs(self.checkpoints_save_path)

    def initialize_loggers(self):
        self.summary_writer = SummaryWriter(os.path.join(self.experiment_path, 'logs'))
        self.train_epoch_logger = Logger(os.path.join(self.experiment_path, 'train_epoch.log'),
                                         ['epoch', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
        self.train_batch_logger = Logger(os.path.join(self.experiment_path, 'train_batch.log'),
                                         ['epoch', 'batch', 'iter', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
        self.validation_logger = Logger(os.path.join(self.experiment_path, 'validation.log'),
                                        ['epoch', 'val_loss', 'val_acc_top1', 'val_acc_top5'])

    def initialize_training(self):
        print('Initializing training ...')

        self.initialize_criterion()
        self.initialize_optimizer()
        self.initialize_scheduler()

        print('\n==========================================')

    def train_model(self, last_epoch=None, checkpoint_path=None):
        if last_epoch is not None:
            print('Resuming {} models training from epoch #{}.'.format(self.model.__class__.__name__, last_epoch))
            self.load_checkpoint(checkpoint_path, last_epoch)
            print('Loaded checkpoint: {}, epoch #{}'.format(checkpoint_path, last_epoch))
        else:
            last_epoch = 1
            if hasattr(self.opts, 'pretrained_weights'):
                print('Training from pretrained weights of {} model.'.format(self.model.__class__.__name__))
                self.model.load_weights(self.opts.pretrained_weights)
                print('Loaded weights: {}'.format(self.opts.pretrained_weights))
            else:
                print('Training {} model from scratch.'.format(self.model.__class__.__name__))

        self.model.to(self.device)
        self.criterion.to(self.device)

        for epoch in range(last_epoch, self.opts.epochs+1):
            self.step_train(epoch, self.train_epoch_logger, self.train_batch_logger)
            self.step_val(epoch, self.validation_logger)

    def load_checkpoint(self, checkpoint_path, last_epoch):
        checkpoint_file = os.path.join(checkpoint_path, CHECKPOINT_FORMAT.format(last_epoch))
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_dict'])
        print('Initialized from checkpoint {}'.format(checkpoint_file))

    def step_train(self, epoch, epoch_logger, batch_logger):
        self.model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        tra_loss = AverageMeter('Tra Loss', ':.4e')
        tra_acc_top1 = AverageMeter('Tra Acc@1', ':6.2f')
        tra_acc_top5 = AverageMeter('Tra Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.dataloader_train),
            [batch_time, data_time, tra_loss, tra_acc_top1, tra_acc_top5],
            prefix="Epoch: [{}]".format(epoch)
        )

        end_time = time.time()
        for idx, (inputs, labels) in enumerate(self.dataloader_train):
            data_time.update(time.time() - end_time)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            outputs = nn.Softmax(dim=1)(outputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            tra_loss.update(loss.item(), inputs.size(0))
            tra_acc_top1.update(acc1[0], inputs.size(0))
            tra_acc_top5.update(acc5[0], inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            # print(self.model.fc_6.weight.grad)
            # print(self.model.fc_7.weight.grad)
            # print(self.model.fc_8.weight.grad)
            self.optimizer.step()
            self.scheduler.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if idx % PRINT_STEP == 0:
                progress.display(idx)

            batch_logger.log({
                'epoch': epoch,
                'batch': idx + 1,
                'iter': (epoch - 1) * len(self.dataloader_train) + (idx + 1),
                'tra_loss': tra_loss.val,
                'tra_acc_top1': tra_acc_top1.val.item(),
                'tra_acc_top5': tra_acc_top5.val.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        epoch_logger.log({
            'epoch': epoch,
            'tra_loss': tra_loss.val,
            'tra_acc_top1': tra_acc_top1.val.item(),
            'tra_acc_top5': tra_acc_top5.val.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        })

        self.summary_writer.add_scalar('data/tra_loss', tra_loss.val, epoch)
        self.summary_writer.add_scalar('data/tra_acc1', tra_acc_top1.val.item(), epoch)
        self.summary_writer.add_scalar('data/tra_acc5', tra_acc_top5.val.item(), epoch)

    def step_val(self, epoch, validation_logger):
        self.model.eval()

        batch_time = AverageMeter('Time', ':6.3f')
        val_loss = AverageMeter('Val Loss', ':.4e')
        val_acc_top1 = AverageMeter('Val Acc@1', ':6.2f')
        val_acc_top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.dataloader_test),
            [batch_time, val_loss, val_acc_top1, val_acc_top5],
            prefix="Validation: Epoch [{}]".format(epoch)
        )

        with torch.no_grad():
            end_time = time.time()
            for idx, (inputs, labels) in enumerate(self.dataloader_test):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                outputs = nn.Softmax(dim=1)(outputs)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                val_loss.update(loss.item(), inputs.size(0))
                val_acc_top1.update(acc1[0], inputs.size(0))
                val_acc_top5.update(acc5[0], inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                progress.display(idx)

            validation_logger.log({
                'epoch': epoch,
                'val_loss': val_loss.val,
                'val_acc_top1': val_acc_top1.val.item(),
                'val_acc_top5': val_acc_top5.val.item()
            })

            self.summary_writer.add_scalar('data/val_loss', val_loss.val, epoch)
            self.summary_writer.add_scalar('data/val_acc1', val_acc_top1.val.item(), epoch)
            self.summary_writer.add_scalar('data/val_acc5', val_acc_top5.val.item(), epoch)

    def initialize_dataloaders(self):
        dataset_train, dataset_test = self.create_datasets()

        print('Initializing data loader for train split: shuffle {}, workers {}'.format(self.opts.shuffle_train, self.opts.num_workers))
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=self.opts.batch_size,
                                           shuffle=self.opts.shuffle_train,
                                           num_workers=self.opts.num_workers)

        print('Initializing data loader for test split: shuffle {}, workers {}'.format(False, self.opts.num_workers))
        self.dataloader_test = DataLoader(dataset_test,
                                          batch_size=self.opts.batch_size,
                                          num_workers=self.opts.num_workers)

        print('\n==========================================')

    def create_datasets(self):
        if self.opts.models[0] == 'c3d':
            dataset_train = DatasetC3D(root_dir=self.opts.dataset_dir,
                                       dataset=self.opts.dataset,
                                       split=SplitType.TRAIN,
                                       modality=self.opts.modality,
                                       sampling=self.opts.sampling,
                                       clip_length=self.opts.clip_length,
                                       transform=self.get_transforms(is_train=True))

            dataset_test = DatasetC3D(root_dir=self.opts.dataset_dir,
                                      dataset=self.opts.dataset,
                                      split=SplitType.TEST,
                                      modality=self.opts.modality,
                                      sampling=self.opts.sampling,
                                      clip_length=self.opts.clip_length,
                                      transform=self.get_transforms(is_train=False))
        else:
            raise NotImplementedError

        return dataset_train, dataset_test

    def get_transforms(self, is_train=True):
        transforms = []
        if hasattr(self.opts, 'random_crop'):
            transforms.append(video_transforms.ClipRandomCrop(size=self.opts.random_crop))
        if self.opts.horizontal_flip and is_train:
            transforms.append(video_transforms.ClipHorizontalFlip())
        transforms.append(video_transforms.ClipToTensor(div=self.opts.normalize))
        if self.opts.standardize_mean and self.opts.standardize_std:
            transforms.append(
                video_transforms.ClipStandardize(mean=self.opts.standardize_mean, std=self.opts.standardize_std))

        return video_transforms.Compose(transforms)

    def initialize_model(self):
        # TODO merging multiple models will be added in the future
        self.model = C3D(num_classes=self.dataloader_train.dataset.num_classes,
                         batch_norm=self.opts.batch_norm,
                         dropout_prob=self.opts.dropout_prob,
                         weight_initializer=self.opts.weight_initializer)
        print('Total number of parameters: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        print('\n==========================================')

    def initialize_criterion(self):
        print('Criterion: {}'.format(self.opts.criterion))
        # TODO multiple losses will be added in the future
        self.criterion = LOSSES[self.opts.criterion]

    def initialize_optimizer(self):
        print('Optimizer: {}'.format(self.opts.optimizer))
        print('\tLearning rate: {}'.format(self.opts.learning_rate))
        if hasattr(self.opts, 'weight_decay'):
            print('\tWeight decay: {}'.format(self.opts.weight_decay))

        # TODO more optimizers will be added soon
        if self.opts.optimizer.lower() == 'sgd':
            print('\tMomentum {}:'.format(self.opts.momentum))
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.opts.learning_rate,
                                       momentum=self.opts.momentum,
                                       weight_decay=self.opts.weight_decay if hasattr(self.opts, 'weight_decay') else 0,
                                       nesterov=True)
        elif self.opts.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.opts.learning_rate,
                                        weight_decay=self.opts.weight_decay if hasattr(self.opts, 'weight_decay') else 0)
        else:
            raise NotImplementedError

    def initialize_scheduler(self):
        if hasattr(self.opts, 'scheduler'):
            print('Learning rate scheduler: {}'.format(self.opts.scheduler))

            # TODO more schedulers will be added soon
            if self.opts.scheduler.lower() == 'step_lr':
                print('\tStep size: {}'.format(self.opts.scheduler_step))
                print('\tMultiplication factor: {}'.format(self.opts.scheduler_factor))
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                           step_size=self.opts.scheduler_step,
                                                           gamma=self.opts.scheduler_factor)
            else:
                raise NotImplementedError
        else:
            print('No scheduler.')
