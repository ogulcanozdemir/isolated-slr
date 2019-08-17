from utils.experiment import Experiment
from utils.parameter_parser import parse_arguments


if __name__ == '__main__':
    params = parse_arguments()

    experiment = Experiment(opts=params)
    if params.last_epoch and params.resume_checkpoint:
        experiment.train_model(last_epoch=params.last_epoch, checkpoint_path=params.resume_checkpoint)
    else:
        experiment.train_model()

    # TODO finish C3D testing script
    # TODO implement model merging (for C3D + LSTM model)
    # TODO implement I3D model
    # TODO implement 3D ResNet Model


    # sgd
    # --learning - rate = 3e-3
    # --weight - decay = 5e-4
    # --scheduler = step_lr
    # --scheduler - step = 150
    # --scheduler - factor = 0.5