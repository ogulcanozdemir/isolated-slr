from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

experiment_folder = '/raid/users/oozdemir/code/untitled-slr-project/experiments/experiment_24.08.2019-19.08.09_bsign_r3d_18_rgb_equidistant_clip16_batch32_adam_cross_entropy_lr0.001'

train_epoch_log = pd.read_csv(os.path.join(experiment_folder, 'train_epoch.log'), sep='\t')
validation_log = pd.read_csv(os.path.join(experiment_folder, 'validation.log'), sep='\t')

merged_df = pd.concat([train_epoch_log, validation_log], axis=1, sort=False)

results_writer_train = SummaryWriter(os.path.join(experiment_folder, 'result_logs/train'))
results_writer_validation = SummaryWriter(os.path.join(experiment_folder, 'result_logs/validation'))

epochs = []
tra_loss = []
val_loss = []
tra_acc_top1 = []
tra_acc_top5 = []
val_acc_top1 = []
val_acc_top5 = []
for index, row in merged_df.iterrows():
    epochs.append(row['epoch'].values[0].astype('int'))
    tra_loss.append(row['tra_loss'])
    val_loss.append(row['val_loss'])
    tra_acc_top1.append(row['tra_acc_top1'])
    tra_acc_top5.append(row['tra_acc_top5'])
    val_acc_top1.append(row['val_acc_top1'])
    val_acc_top5.append(row['val_acc_top5'])

    results_writer_train.add_scalar('loss', row['tra_loss'], row['epoch'].values[0].astype('int'))
    results_writer_train.add_scalar('accuracy', row['tra_acc_top1'], row['epoch'].values[0].astype('int'))
    results_writer_train.add_scalar('accuracy_top5', row['tra_acc_top5'], row['epoch'].values[0].astype('int'))

    results_writer_validation.add_scalar('loss', row['val_loss'], row['epoch'].values[0].astype('int'))
    results_writer_validation.add_scalar('accuracy', row['val_acc_top1'], row['epoch'].values[0].astype('int'))
    results_writer_validation.add_scalar('accuracy_top5', row['val_acc_top5'], row['epoch'].values[0].astype('int'))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(epochs, tra_loss, label='training', linewidth=2)
ax1.plot(epochs, val_loss, label='validation', linewidth=2)
ax1.legend()
ax1.set_title('Loss')

ax2.plot(epochs, tra_acc_top1, label='training', linewidth=2)
ax2.plot(epochs, val_acc_top1, label='validation', linewidth=2)
ax2.legend()
ax2.set_title('Accuracy - max %{:.2f} at epoch #{}'.format(np.max(val_acc_top1), np.argmax(val_acc_top1)))

ax3.plot(epochs, tra_acc_top5, label='training', linewidth=2)
ax3.plot(epochs, val_acc_top5, label='validation', linewidth=2)
ax3.legend()
ax3.set_title('Accuracy Top_5 - max %{:.2f} at epoch #{}'.format(np.max(val_acc_top5), np.argmax(val_acc_top5)))

f.savefig(os.path.join(experiment_folder, 'results.jpg'), bbox_inches='tight')

print('done')