from utils.functions import AverageMeter, ProgressMeter, CsvLogger, accuracy, get_device, plot_confusion_matrix
from utils.logger import Logger, logger
from utils.constants import *
from utils import video_transforms
from utils import sequence_transforms
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

from datasets.dataset_c3d import DatasetC3D
from datasets.dataset_lstm import DatasetLSTM, PadCollate
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from tensorboardX import SummaryWriter
from models.C3D import C3D
from models.LSTM import MultiLayerLSTM

import matplotlib.pyplot as plt
import time
import os
import torch
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import operator


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

    def create_experiment_folders(self):
        # FIXME will be changed (related to multiple models)
        self.experiment_date = time.strftime("%d.%m.%Y-%H.%M.%S")
        self.experiment_id = 'experiment_' \
                             + self.experiment_date + '_' \
                             + self.opts.dataset + '_' \
                             + self.opts.models[0] + '_' \
                             + self.opts.modality + '_' \
                             + self.opts.sampling + '_' \
                             + ('clip' + str(self.opts.clip_length) if self.opts.clip_length else ('seq' + str(self.opts.seq_length))) + '_' \
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

    def __init__(self, opts):
        self.opts = opts

        self.device = get_device()
        if not self.opts.experiment_path:
            self.create_experiment_folders()
        else:
            self.experiment_path = self.opts.experiment_path
            self.checkpoints_save_path = os.path.join(self.experiment_path, 'checkpoints')

        self.initialize_loggers()

        self.initialize_dataloaders()
        self.initialize_model()

        if not self.opts.experiment_path:
            self.initialize_training()

    def initialize_loggers(self):
        if self.opts.test_mode == 'off':
            self.summary_writer_train = SummaryWriter(os.path.join(self.experiment_path, 'logs', 'train'))
            self.summary_writer_validation = SummaryWriter(os.path.join(self.experiment_path, 'logs', 'validation'))
            self.train_epoch_logger = CsvLogger(os.path.join(self.experiment_path, 'train_epoch.log'),
                                                ['epoch', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
            # self.train_batch_logger = CsvLogger(os.path.join(self.experiment_path, 'train_batch.log'),
            #                                     ['phase', 'epoch', 'batch', 'iter', 'tra_loss', 'tra_acc_top1', 'tra_acc_top5', 'lr'])
            self.validation_logger = CsvLogger(os.path.join(self.experiment_path, 'validation.log'),
                                               ['epoch', 'val_loss', 'val_acc_top1', 'val_acc_top5'])

            Logger.__call__().add_file_handler(self.opts.log_level, os.path.join(self.experiment_path, 'experiment.log'))
        else:
            Logger.__call__().add_file_handler(self.opts.log_level, os.path.join(self.experiment_path, 'test.log'))

        Logger.__call__().set_log_level(log_level=self.opts.log_level)
        Logger.__call__().add_stream_handler(log_level=self.opts.log_level)

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
                logger.info('Training {} model.'.format(self.model.__class__.__name__))

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
        if len(self.opts.models) == 0:
            raise AssertionError

        if self.opts.models[0] in ['c3d', 'r3d_18', 'mc3_18', 'r2plus1d_18']:
            if not self.opts.experiment_path:
                dataset_train = DatasetC3D(root_dir=self.opts.dataset_dir,
                                           dataset=self.opts.dataset,
                                           split=SplitType.TRAIN,
                                           modality=self.opts.modality,
                                           sampling=self.opts.sampling,
                                           frame_size=self.opts.frame_size,
                                           clip_length=self.opts.clip_length,
                                           transform=self.get_transforms(is_train=True))

                logger.info('Initializing data loader for train split: batch size {}, shuffle {}, workers {}'.format(self.opts.batch_size, self.opts.shuffle_train, self.opts.num_workers))
                self.dataloaders[SplitType.TRAIN] = DataLoader(dataset_train,
                                                               batch_size=self.opts.batch_size,
                                                               shuffle=self.opts.shuffle_train,
                                                               num_workers=self.opts.num_workers)

            if self.opts.feature_layer:
                dataset_train = DatasetC3D(root_dir=self.opts.dataset_dir,
                                           dataset=self.opts.dataset,
                                           split=SplitType.TRAIN,
                                           modality=self.opts.modality,
                                           sampling=self.opts.sampling,
                                           frame_size=self.opts.frame_size,
                                           clip_length=self.opts.clip_length,
                                           transform=self.get_transforms(is_train=False),
                                           test_mode=True if self.opts.test_mode == 'on' else False,
                                           feature_extract=True)

                logger.info('Initializing data loader for train split: batch size {}, shuffle {}, workers {}'.format(self.opts.batch_size, self.opts.shuffle_train, self.opts.num_workers))
                self.dataloaders[SplitType.TRAIN] = DataLoader(dataset_train,
                                                               batch_size=1 if self.opts.experiment_path and self.opts.test_mode == 'on' else self.opts.batch_size,
                                                               num_workers=self.opts.num_workers)

            dataset_validation = DatasetC3D(root_dir=self.opts.dataset_dir,
                                            dataset=self.opts.dataset,
                                            split=SplitType.TEST,
                                            modality=self.opts.modality,
                                            sampling=self.opts.sampling,
                                            frame_size=self.opts.frame_size,
                                            clip_length=self.opts.clip_length,
                                            transform=self.get_transforms(is_train=False),
                                            test_mode=True if self.opts.test_mode == 'on' else False,
                                            feature_extract=True if self.opts.feature_layer else False)

            logger.info('Initializing data loader for validation split: batch size {}, workers {}'.format(self.opts.batch_size, self.opts.num_workers))
            self.dataloaders[SplitType.VAL] = DataLoader(dataset_validation,
                                                         batch_size=1 if self.opts.experiment_path and self.opts.test_mode == 'on' else self.opts.batch_size,
                                                         num_workers=self.opts.num_workers)
        elif self.opts.models[0] in ['rnn', 'lstm', 'gru']:
            if not self.opts.experiment_path:
                dataset_train = DatasetLSTM(root_dir=self.opts.dataset_dir,
                                            dataset=self.opts.dataset,
                                            split=SplitType.TRAIN,
                                            modality=self.opts.modality,
                                            sequence_len=self.opts.seq_length,
                                            sampling=self.opts.sampling,
                                            transform=self.get_transforms(is_train=True))

                logger.info('Initializing data loader for train split: batch size {}, shuffle {}, workers {}'.format(self.opts.batch_size, self.opts.shuffle_train, self.opts.num_workers))
                self.dataloaders[SplitType.TRAIN] = DataLoader(dataset_train,
                                                               batch_size=self.opts.batch_size,
                                                               shuffle=self.opts.shuffle_train,
                                                               num_workers=self.opts.num_workers,
                                                               collate_fn=PadCollate(dim=0))

            dataset_validation = DatasetLSTM(root_dir=self.opts.dataset_dir,
                                             dataset=self.opts.dataset,
                                             split=SplitType.TEST,
                                             modality=self.opts.modality,
                                             sequence_len=self.opts.seq_length,
                                             sampling=self.opts.sampling,
                                             transform=self.get_transforms(is_train=False),
                                             test_mode=True if self.opts.test_mode == 'on' else False)

            logger.info('Initializing data loader for validation split: batch size {}, workers {}'.format(self.opts.batch_size, self.opts.num_workers))
            self.dataloaders[SplitType.VAL] = DataLoader(dataset_validation,
                                                         batch_size=1 if self.opts.experiment_path and self.opts.test_mode == 'on' else self.opts.batch_size,
                                                         num_workers=self.opts.num_workers,
                                                         collate_fn=PadCollate(dim=0))

        logger.info('==========================================')

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

        if self.opts.clip_length:
            transforms.append(video_transforms.ClipToTensor())
        if self.opts.seq_length:
            transforms.append(sequence_transforms.ClipToTensor())

        if self.opts.standardize_mean and self.opts.standardize_std:
            transforms.append(video_transforms.ClipStandardize(mean=self.opts.standardize_mean, std=self.opts.standardize_std))

        return video_transforms.Compose(transforms)

    def initialize_model(self):
        # TODO merging multiple models will be added in the future
        if self.opts.models[0] == 'c3d':
            self.model = C3D(num_classes=list(self.dataloaders.items())[0][1].dataset.num_classes,
                             batch_norm=self.opts.batch_norm,
                             dropout_prob=self.opts.dropout_prob,
                             weight_initializer=self.opts.weight_initializer)

        elif self.opts.models[0] == 'r3d_18':
            self.model = r3d_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, list(self.dataloaders.items())[0][1].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        elif self.opts.models[0] == 'mc3_18':
            self.model = mc3_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, list(self.dataloaders.items())[0][1].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        elif self.opts.models[0] == 'r2plus1d_18':
            self.model = r2plus1d_18(pretrained=self.opts.pretrained)
            self.model.fc = nn.Linear(512, list(self.dataloaders.items())[0][1].dataset.num_classes, bias=True)
            torch.nn.init.xavier_normal_(self.model.fc.weight)
            self.model.fc.bias.data.fill_(1)

        elif self.opts.models[0] in ['rnn', 'lstm', 'gru']:
            layer_dict = {
                'rnn': nn.RNN,
                'lstm': nn.LSTM,
                'gru': nn.GRU
            }

            self.model = MultiLayerLSTM(512,
                                        layer_type=layer_dict[self.opts.models[0]],
                                        num_hidden=self.opts.num_rnn_hidden,
                                        batch_first=True,
                                        num_layers=self.opts.num_rnn_layers,
                                        dropout=self.opts.dropout_prob,
                                        bidirectional=self.opts.bidirectional,
                                        device=self.device)

            classification_layer = nn.Linear(self.opts.num_rnn_hidden * (2 if self.opts.bidirectional else 1),
                                             list(self.dataloaders.items())[0][1].dataset.num_classes,
                                             bias=True)
            torch.nn.init.xavier_normal_(classification_layer.weight)
            classification_layer.bias.data.fill_(1)
            self.model.fc = classification_layer

        if not self.opts.experiment_path:
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

    def test_model(self):
        use_split = SplitType.VAL
        # sort checkpoints by date modified, and get the best model
        checkpoints_list = glob.glob(os.path.join(self.checkpoints_save_path, '*.pth'))
        checkpoints_list.sort(key=os.path.getmtime)
        checkpoint_file = os.path.join(self.checkpoints_save_path, checkpoints_list[-1])
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Initialized from checkpoint {}'.format(checkpoint_file))

        self.model.to(self.device)
        self.model.eval()

        acc_top1 = AverageMeter('acc@1', ':.2f')
        acc_top5 = AverageMeter('acc@5', ':.2f')

        plot_labels = []
        plot_predictions = []

        file_info = {}
        for idx, (inputs, labels, names) in enumerate(self.dataloaders[use_split]):
            cls, video = (names[0].split(os.sep)[-2], names[0].split(os.sep)[-1])
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc_top1.update(acc1[0], inputs.size(0))
            acc_top5.update(acc5[0], inputs.size(0))

            # get predictions
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            org_lbl = self.dataloaders[use_split].dataset.label2idx[str(labels.item())]
            pred_lbls = [self.dataloaders[use_split].dataset.label2idx[str(p.item())] for p in pred]

            file_info[str(cls) + os.sep + video] = {'label': org_lbl, 'prediction': pred_lbls}

            logger.info('({}/{}) - {}, {} : label {}, predictions {}'.format(idx+1,
                                                     len(self.dataloaders[use_split]),
                                                     cls,
                                                     video,
                                                     org_lbl,
                                                     pred_lbls))
            plot_labels.append(org_lbl[1])
            plot_predictions.append(pred_lbls[0][1])

        logger.info('Accuracy {:.2f}'.format(acc_top1.avg.item()))
        logger.info('Accuracy Top5 {:.2f}'.format(acc_top5.avg.item()))

        if self.opts.plot_confusion:
            _, cm = plot_confusion_matrix(plot_labels, plot_predictions, unique_labels(plot_labels, plot_predictions))
            plt.savefig(os.path.join(self.experiment_path, 'confusion_matrix.png'))

        np.savez(os.path.join(self.experiment_path, 'predictions.npz'), file_info)
        analyze_results(experiment_path=self.experiment_path)

    def extract_features(self, split):
        use_split = split
        # sort checkpoints by date modified, and get the best model
        checkpoints_list = glob.glob(os.path.join(self.checkpoints_save_path, '*.pth'))
        checkpoints_list.sort(key=os.path.getmtime)
        checkpoint_file = os.path.join(self.checkpoints_save_path, checkpoints_list[-1])
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Initialized from checkpoint {}'.format(checkpoint_file))

        feature_layer = self.model._modules.get(self.opts.feature_layer)  # for example avgpool

        self.model.to(self.device)
        self.model.eval()

        for idx, (inputs, names) in enumerate(self.dataloaders[use_split]):
            cls, video = (names[0].split(os.sep)[-2], names[0].split(os.sep)[-1])

            logger.info('Extracting feature ({}/{}) - {}, {}'.format(idx+1,
                                                                     len(self.dataloaders[use_split]),
                                                                     cls,
                                                                     video))

            features = []
            for input in inputs:
                model_input = input.to(self.device)
                feature = torch.zeros(1, 512)
                with torch.set_grad_enabled(False):
                    # define a hook function
                    def copy_data(m, i, o):
                        feature.copy_(o.flatten(1).data)
    
                    h = feature_layer.register_forward_hook(copy_data)
                    self.model(model_input)
                    h.remove()
                    
                features.append(feature.cpu().numpy()[0])

            feature_cls_path = os.path.join(self.opts.feature_path, cls)
            if not os.path.exists(feature_cls_path):
                os.makedirs(feature_cls_path)

            np.savez(os.path.join(feature_cls_path, video + '.npz'), np.array(features))


def analyze_results(experiment_path):
    predictions = np.load(os.path.join(experiment_path, 'predictions.npz'), allow_pickle=True)['arr_0'].item()

    target_labels = []
    predicted_labels = []
    for video in predictions:
        video_preds = predictions[video]
        target_labels.append(video_preds['label'][1])
        predicted_labels.append(video_preds['prediction'][0][1])

    labels = sorted(unique_labels(target_labels, predicted_labels))

    cm = confusion_matrix(target_labels, predicted_labels, labels=labels)
    fig = print_confusion_matrix(cm, labels, figsize=(20, 20), fontsize=20, print_labels=False)
    plt.savefig(os.path.join(experiment_path, 'confusion_matrix.png'), dpi=200)

    class_acc = {}
    for idx, lbl in enumerate(labels):
        acc = np.round(100 * (cm[idx, idx] / sum(cm[idx, :])), decimals=2)
        print('{} : {:.2f} - size {} - correct {} - '.format(labels[idx],
                                                             acc,
                                                             sum(cm[idx, :]),
                                                             cm[idx, idx]), end='')
        confused_indices = np.nonzero(cm[idx, :])[0]

        confused_sample_siz = {}
        for c_idx in confused_indices:
            if c_idx == idx:
                continue
            confused_sample_siz[labels[c_idx]] = cm[idx, c_idx]
        sorted_confused_sample_siz = sorted(confused_sample_siz.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_confused_sample_siz)

        if len(sorted_confused_sample_siz) == 0:
            class_acc[labels[idx]] = (acc, {'size': sum(cm[idx, :]),
                                            'correct': cm[idx, idx]})
        else:
            class_acc[labels[idx]] = (acc, {'size': sum(cm[idx, :]),
                                            'correct': cm[idx, idx],
                                            'confused_samples': sorted_confused_sample_siz})

    with open(os.path.join(experiment_path, 'predictions.txt'), 'w', encoding='utf8') as f:
        sorted_class_acc = sorted(class_acc.items(), key=lambda item: item[1][0], reverse=True)
        for k, v in sorted_class_acc:
            print(k, v, file=f)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14, print_labels=True):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=False, fmt="d", xticklabels=print_labels, yticklabels=print_labels,
                              cbar=False, cmap=sns.cm.rocket_r)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    if print_labels:
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    return fig