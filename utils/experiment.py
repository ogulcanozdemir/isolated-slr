from utils.functions import AverageMeter, ProgressMeter, CsvLogger, accuracy, get_device
from utils.logger import Logger, logger
from utils.constants import *
from utils import video_transforms

from datasets.dataset_c3d import DatasetC3D
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from tensorboardX import SummaryWriter
from models.C3D import C3D

import time
import os
import torch


class Experiment:

    experiment_date = None
    experiment_id = None
    experiment_path = None
    checkpoints_save_path = None
    summary_writer_train = None
    summary_writer_validation = None
    train_epoch_logger = None
    train_batch_logger = None
    validation_logger = None
    logger = None

    dataloaders = {}
    model = None
    optimizer = None
    criterion = None
    scheduler = None

    best_accuracy = 0.0

    def __init__(self, opts):
        self.opts = opts

        self.create_experiment_folders()
        self.initialize_loggers()

        self.device = get_device()
        self.initialize_dataloaders()
        self.initialize_model()
        self.initialize_training()

    def create_experiment_folders(self):
        # FIXME will be changed (related to multiple models)
        self.experiment_date = time.strftime("%d.%m.%Y-%H.%M.%S")
        self.experiment_id = 'experiment_' \
                             + self.experiment_date + '_' \
                             + self.opts.dataset + '_' \
                             + self.opts.models[0] + '_' \
                             + self.opts.modality + '_' \
                             + self.opts.sampling + '_' \
                             + 'clip' + str(self.opts.clip_length) + '_' \
                             + 'batch' + str(self.opts.batch_size) + '_' \
                             + self.opts.optimizer + '_' \
                             + self.opts.criterion + '_' \
                             + 'lr' + str(self.opts.learning_rate)

        self.experiment_path = os.path.join(EXPERIMENT_ROOT, self.experiment_id)
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        else:
            raise FileExistsError

        self.checkpoints_save_path = os.path.join(self.experiment_path, 'checkpoints')
        os.makedirs(self.checkpoints_save_path)

    def initialize_loggers(self):
        self.summary_writer_train = SummaryWriter(os.path.join(self.experiment_path, 'logs', 'train'))
        self.summary_writer_validation = SummaryWriter(os.path.join(self.experiment_path, 'logs', 'validation'))
        self.train_epoch_logger = CsvLogger(os.path.join(self.experiment_path, 'train_epoch.log'),
                                            ['epoch', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
        # self.train_batch_logger = CsvLogger(os.path.join(self.experiment_path, 'train_batch.log'),
        #                                     ['phase', 'epoch', 'batch', 'iter', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
        self.validation_logger = CsvLogger(os.path.join(self.experiment_path, 'validation.log'),
                                           ['epoch', 'val_loss', 'val_acc_top1', 'val_acc_top5'])

        Logger.__call__().set_log_level(log_level=self.opts.log_level)
        Logger.__call__().add_stream_handler(log_level=self.opts.log_level)
        Logger.__call__().add_file_handler(self.opts.log_level, os.path.join(self.experiment_path, 'experiment.log'))

    def initialize_training(self):
        logger.info('Initializing training ...')

        self.initialize_criterion()
        self.initialize_optimizer()
        self.initialize_scheduler()

        logger.info('==========================================')

    def train_model(self, last_epoch=None, checkpoint_path=None):
        if last_epoch is not None:
            logger.info('Resuming {} models training from epoch #{}.'.format(self.model.__class__.__name__, last_epoch))
            self.load_checkpoint(checkpoint_path, last_epoch)
            logger.info('Loaded checkpoint: {}, epoch #{}'.format(checkpoint_path, last_epoch))
        else:
            last_epoch = 1
            if self.opts.pretrained and self.opts.pretrained_weights:
                logger.info('Training from pretrained weights of {} model.'.format(self.model.__class__.__name__))
                self.model.load_weights(self.opts.pretrained_weights)
                logger.info('Loaded weights: {}'.format(self.opts.pretrained_weights))
            else:
                logger.info('Training {} model from scratch.'.format(self.model.__class__.__name__))

        self.model.to(self.device)
        self.criterion.to(self.device)

        for epoch in range(last_epoch, self.opts.epochs+1):
            # self.step_train(epoch, self.train_epoch_logger, self.train_batch_logger)
            # self.step_val(epoch, self.validation_logger)
            self.step_train_val(epoch, self.train_epoch_logger, self.validation_logger)

    def load_checkpoint(self, checkpoint_path, last_epoch):
        checkpoint_file = os.path.join(checkpoint_path, CHECKPOINT_FORMAT.format(last_epoch))
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Initialized from checkpoint {}'.format(checkpoint_file))

    def initialize_dataloaders(self):
        dataset_train, dataset_validation = self.create_datasets()

        logger.info('Initializing data loader for train split: batch size {}, shuffle {}, workers {}'.format(self.opts.batch_size, self.opts.shuffle_train, self.opts.num_workers))
        self.dataloaders[SplitType.TRAIN] = DataLoader(dataset_train,
                                                       batch_size=self.opts.batch_size,
                                                       shuffle=self.opts.shuffle_train,
                                                       num_workers=self.opts.num_workers)

        logger.info('Initializing data loader for validation split: batch size {}, workers {}'.format(self.opts.batch_size, self.opts.num_workers))
        self.dataloaders[SplitType.VAL] = DataLoader(dataset_validation,
                                                     batch_size=self.opts.batch_size,
                                                     num_workers=self.opts.num_workers)

        logger.info('==========================================')

    def create_datasets(self):
        if len(self.opts.models) == 0:
            raise AssertionError

        if self.opts.models[0] in ['c3d', 'r3d_18', 'mc3_18', 'r2plus1d_18']:
            dataset_train = DatasetC3D(root_dir=self.opts.dataset_dir,
                                       dataset=self.opts.dataset,
                                       split=SplitType.TRAIN,
                                       modality=self.opts.modality,
                                       sampling=self.opts.sampling,
                                       frame_size=self.opts.frame_size,
                                       clip_length=self.opts.clip_length,
                                       transform=self.get_transforms(is_train=True))

            dataset_validation = DatasetC3D(root_dir=self.opts.dataset_dir,
                                            dataset=self.opts.dataset,
                                            split=SplitType.TEST,
                                            modality=self.opts.modality,
                                            sampling=self.opts.sampling,
                                            frame_size=self.opts.frame_size,
                                            clip_length=self.opts.clip_length,
                                            transform=self.get_transforms(is_train=False))
        else:
            raise NotImplementedError

        return dataset_train, dataset_validation

    def get_transforms(self, is_train=True):
        transforms = []
        if self.opts.random_crop:
            transforms.append(video_transforms.ClipRandomCrop(size=self.opts.random_crop))
        if self.opts.crop_mean:
            transforms.append(video_transforms.ClipSubtractMean(crop_mean=self.opts.crop_mean))
        if self.opts.horizontal_flip and is_train:
            transforms.append(video_transforms.ClipHorizontalFlip())
        if self.opts.normalize:
            transforms.append(video_transforms.ClipNormalize(clip_range=tuple(self.opts.normalize)))
        transforms.append(video_transforms.ClipToTensor())
        if self.opts.standardize_mean and self.opts.standardize_std:
            transforms.append(
                video_transforms.ClipStandardize(mean=self.opts.standardize_mean, std=self.opts.standardize_std))

        return video_transforms.Compose(transforms)

    def initialize_model(self):
        # TODO merging multiple models will be added in the future
        if self.opts.models[0] == 'c3d':
            self.model = C3D(num_classes=self.dataloaders[SplitType.TRAIN].dataset.num_classes,
                             batch_norm=self.opts.batch_norm,
                             dropout_prob=self.opts.dropout_prob,
                             weight_initializer=self.opts.weight_initializer)

        elif self.opts.models[0] == 'r3d_18':
            self.model = r3d_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, self.dataloaders[SplitType.TRAIN].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        elif self.opts.models[0] == 'mc3_18':
            self.model = mc3_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, self.dataloaders[SplitType.TRAIN].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        elif self.opts.models[0] == 'r2plus1d_18':
            self.model = r2plus1d_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, self.dataloaders[SplitType.TRAIN].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        if len(self.opts.layers) > 0 and self.opts.pretrained:
            logger.info('Finetuning layers: {}'.format(self.opts.layers))
            for name, child in self.model.named_children():
                if name in self.opts.layers:
                    # logger.info(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    # logger.info(name + ' is frozen')
                    for param in child.parameters():
                        param.requires_grad = False
        elif self.opts.pretrained:
            logger.info('Finetuning all layers.')
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True
        else:
            logger.info('Training all layers from scratch.')
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

        logger.info('Total number of parameters: %.2fM' % (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1000000.0))
        logger.info('==========================================')

    def initialize_criterion(self):
        logger.info('Criterion: {}'.format(self.opts.criterion))
        # TODO multiple losses will be added in the future
        self.criterion = LOSSES[self.opts.criterion]

    def initialize_optimizer(self):
        logger.info('Optimizer: {}'.format(self.opts.optimizer))
        logger.info('\tLearning rate: {}'.format(self.opts.learning_rate))
        if self.opts.weight_decay:
            logger.info('\tWeight decay: {}'.format(self.opts.weight_decay))

        # TODO more optimizers will be added soon
        if self.opts.optimizer.lower() == 'sgd':
            if self.opts.momentum:
                logger.info('\tMomentum: {}'.format(self.opts.momentum))
            else:
                self.opts.momentum = 0.9
                logger.info('\tMomentum: Not specified, using {}'.format(self.opts.momentum))

            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.opts.learning_rate,
                                       momentum=self.opts.momentum,
                                       weight_decay=self.opts.weight_decay if self.opts.weight_decay else 0,
                                       nesterov=True)
        elif self.opts.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.opts.learning_rate,
                                        weight_decay=self.opts.weight_decay if self.opts.weight_decay else 0)
        else:
            raise NotImplementedError

    def initialize_scheduler(self):
        if self.opts.scheduler:
            logger.info('Learning rate scheduler: {}'.format(self.opts.scheduler))

            # TODO more schedulers will be added soon
            if self.opts.scheduler.lower() == 'step_lr':
                logger.info('\tStep size: {}'.format(self.opts.scheduler_step))
                logger.info('\tMultiplication factor: {}'.format(self.opts.scheduler_factor))
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                           step_size=self.opts.scheduler_step,
                                                           gamma=self.opts.scheduler_factor)
            else:
                raise NotImplementedError
        else:
            logger.info('No scheduler.')

    def step_train_val(self, epoch, epoch_logger, validation_logger):
        for phase in [SplitType.TRAIN, SplitType.VAL]:
            if phase == SplitType.TRAIN:
                self.model.train()
            else:
                self.model.eval()

            batch_time = AverageMeter('time', ':.3f')
            data_time = AverageMeter('data', ':.3f')
            running_loss = AverageMeter('loss', ':.4f')
            acc_top1 = AverageMeter('acc@1', ':.2f')
            acc_top5 = AverageMeter('acc@5', ':.2f')
            progress = ProgressMeter(
                len(self.dataloaders[phase]),
                [batch_time, data_time, running_loss, acc_top1, acc_top5],
                prefix="Epoch: [{}] {}".format(epoch, phase.value),
            )

            end_time = time.time()
            # first_batch = next(iter(self.dataloaders[phase]))
            for idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                data_time.update(time.time() - end_time)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == SplitType.TRAIN):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    outputs = nn.Softmax(dim=1)(outputs)

                    if phase == SplitType.TRAIN:
                        loss.backward()
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_loss.update(loss.item(), inputs.size(0))
                acc_top1.update(acc1[0], inputs.size(0))
                acc_top5.update(acc5[0], inputs.size(0))
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                # batch_logger.log({
                #     'phase': phase,
                #     'epoch': epoch,
                #     'batch': idx + 1,
                #     'iter': (epoch - 1) * len(self.dataloaders[phase]) + (idx + 1),
                #     'tra_loss': loss.val,
                #     'tra_acc_top1': acc_top1.val.item(),
                #     'tra_acc_top5': acc_top5.val.item(),
                #     'lr': self.optimizer.param_groups[0]['lr']
                # })

            if epoch % PRINT_STEP == 0:
                progress.display(epoch)

            if phase == SplitType.TRAIN:
                self.summary_writer_train.add_scalar('loss', running_loss.avg, epoch)
                self.summary_writer_train.add_scalar('accuracy', acc_top1.avg.item(), epoch)
                self.summary_writer_train.add_scalar('accuracy_top5', acc_top5.avg.item(), epoch)

                epoch_logger.log({
                    'epoch': epoch,
                    'tra_loss': running_loss.avg,
                    'tra_acc_top1': acc_top1.avg.item(),
                    'tra_acc_top5': acc_top5.avg.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            else:
                self.summary_writer_validation.add_scalar('loss', running_loss.avg, epoch)
                self.summary_writer_validation.add_scalar('accuracy', acc_top1.avg.item(), epoch)
                self.summary_writer_validation.add_scalar('accuracy_top5', acc_top5.avg.item(), epoch)

                validation_logger.log({
                    'epoch': epoch,
                    'val_loss': running_loss.avg,
                    'val_acc_top1': acc_top1.avg.item(),
                    'val_acc_top5': acc_top5.avg.item()
                })

                if acc_top1.avg.item() > self.best_accuracy:
                    save_file = 'model_epoch{}_loss{:.4f}_acc{:.2f}.pth'.format(epoch, running_loss.avg,
                                                                                acc_top1.avg.item())
                    logger.info(
                        'Accuracy improved from {:.2f} to {:.2f} , saving model to {}'.format(
                            self.best_accuracy,
                            acc_top1.avg.item(),
                            save_file))

                    # TODO not working as intended
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict,
                                'acc1': acc_top1.avg.item(),
                                'acc5': acc_top5.avg.item(),
                                'loss': running_loss.avg},
                               os.path.join(self.checkpoints_save_path, save_file))
                    self.best_accuracy = acc_top1.avg.item()
