from utils.constants import *

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='C3D Training')
    parser.add_argument('--dataset',
                        action='store', dest='dataset', type=str,
                        help='Dataset Name')
    parser.add_argument('--dataset-dir',
                        action='store', dest='dataset_dir', type=str,
                        help='Directory of dataset')
    parser.add_argument('--shuffle-train',
                        action='store_true', dest='shuffle_train',
                        help='Shuffle train')
    parser.add_argument('--models', choices=['c3d', 'r3d_18', 'mc3_18', 'r2plus1d_18', 'rnn', 'lstm', 'gru'],
                        action='store', nargs='+', default=[], dest='models',
                        help='List of models that will be concatenated')
    parser.add_argument('--layers',
                        action='store', nargs='+', default=[], dest='layers',
                        help='List of models that will be concatenated')
    parser.add_argument('--num-rnn-layers',
                        action='store', dest='num_rnn_layers', type=int,
                        help='number of stacked rnn layer')
    parser.add_argument('--num-rnn-hidden',
                        action='store', dest='num_rnn_hidden', type=int,
                        help='number of stacked rnn hidden units')
    parser.add_argument('--bidirectional',
                        action='store_true', dest='bidirectional',
                        help='for Bidirectional model training')
    parser.add_argument('--seq-length',
                        action='store', dest='seq_length', type=int,
                        help='Sequence length')
    parser.add_argument('--input-dim',
                        action='store', dest='input_dim', type=int,
                        help='Input dimensionality')
    parser.add_argument('--modality', choices=[e.value for e in InputType], default=InputType.RGB.value,
                        action='store', dest='modality', type=str,
                        help='Input modality')
    parser.add_argument('--sampling', choices=[e.value for e in SamplingType], default=SamplingType.RANDOM.value,
                        action='store', dest='sampling', type=str,
                        help='Input sampling')
    parser.add_argument('--clip-length',
                        action='store', dest='clip_length', type=int,
                        help='Clip length')
    # parser.add_argument('--logging',
    #                     action='store_true', dest='logging',
    #                     help='Enable logging')
    parser.add_argument('--log-level', choices=['debug', 'info', 'error'], default='info',
                        action='store', dest='log_level', type=str,
                        help='Log level')

    # data augmentation parameters
    parser.add_argument('--random-crop',
                        action='store', dest='random_crop', type=int,
                        help='Random crop size')
    parser.add_argument('--resize',
                        action='store', dest='resize', type=int,
                        help='Resize frames to size')
    parser.add_argument('--frame-size',
                        action='store', dest='frame_size', type=str,
                        help='Frame size')
    parser.add_argument('--horizontal-flip',
                        action='store_true', dest='horizontal_flip',
                        help='Horizontal flip')
    parser.add_argument('--normalize',
                        action='store', nargs=2, dest='normalize', type=int,
                        help='Input normalization')
    parser.add_argument('--standardize-mean',
                        action='store', nargs=3, dest='standardize_mean', type=float,
                        help='Subtract Mean')
    parser.add_argument('--standardize-std',
                        action='store', nargs=3, dest='standardize_std', type=float,
                        help='Subtract Std')
    parser.add_argument('--crop-mean',
                        action='store', dest='crop_mean', type=str,
                        help='Crop mean file')

    # training parameters
    parser.add_argument('--batch-size', default=10,
                        action='store', dest='batch_size', type=int,
                        help='Batch size')
    parser.add_argument('--num-workers', default=4,
                        action='store', dest='num_workers', type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--criterion', choices=list(LOSSES.keys()), default='cross_entropy',
                        action='store', dest='criterion', type=str,
                        help='Loss function name')
    parser.add_argument('--optimizer', choices=list(OPTIMIZERS.keys()), default='sgd',
                        action='store', dest='optimizer', type=str,
                        help='Network optimizer')
    parser.add_argument('--learning-rate', default=1e-3,
                        action='store', dest='learning_rate', type=float,
                        help='Learning rate')
    parser.add_argument('--momentum',
                        action='store', dest='momentum', type=float,
                        help='SGD momentum')
    parser.add_argument('--weight-decay',
                        action='store', dest='weight_decay', type=float,
                        help='Weight_decay')
    parser.add_argument('--scheduler', choices=list(SCHEDULERS.keys()),
                        action='store', dest='scheduler', type=str,
                        help='Learning rate scheduler ')
    parser.add_argument('--scheduler-step',
                        action='store', dest='scheduler_step', type=int,
                        help='Learning rate scheduler step size')
    parser.add_argument('--scheduler-factor',
                        action='store', dest='scheduler_factor', type=float,
                        help='Learning rate scheduler multiplication factor')
    parser.add_argument('--epoch', default=100,
                        action='store', dest='epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--pretrained-weights',
                        action='store', dest='pretrained_weights', type=str,
                        help='Pretrained weights path')
    parser.add_argument('--pretrained',
                        action='store_true', dest='pretrained',
                        help='use pretrained model when true')
    parser.add_argument('--resume-checkpoint',
                        action='store', dest='resume_checkpoint', type=str,
                        help='Checkpoint load path')
    parser.add_argument('--last-epoch',
                        action='store', dest='last_epoch', type=int,
                        help='Checkpoint load epoch id')

    # model parameters
    parser.add_argument('--batch-norm',
                        action='store_true', dest='batch_norm',
                        help='batch norm if true')
    parser.add_argument('--dropout-prob',
                        action='store', dest='dropout_prob', type=float, default=0,
                        help='dropout is added if set')
    parser.add_argument('--weight-initializer', choices=list(WEIGHT_INIT.keys()), default='xavier_normal',
                        action='store', dest='weight_initializer', type=str,
                        help='Weight initializer for layers')

    # test parameters
    parser.add_argument('--test-mode', choices=['on', 'off'], default='off',
                        action='store', dest='test_mode', type=str,
                        help='test mode flag')
    parser.add_argument('--experiment-path',
                        action='store', dest='experiment_path', type=str,
                        help='experiment path to load')
    parser.add_argument('--plot-confusion',
                        action='store_true', dest='plot_confusion',
                        help='use pretrained model when true')

    # feature extraction parameters
    parser.add_argument('--feature-path',
                        action='store', dest='feature_path', type=str,
                        help='feature extraction path')
    parser.add_argument('--feature-layer',
                        action='store', dest='feature_layer', type=str,
                        help='feature layer')

    return parser.parse_args()