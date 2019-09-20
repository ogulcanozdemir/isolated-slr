from utils.experiment import Experiment
from utils.parameter_parser import parse_arguments
from utils.constants import SplitType

if __name__ == '__main__':
    params = parse_arguments()

    experiment = Experiment(opts=params)
    experiment.extract_features(SplitType.TRAIN)
    experiment.extract_features(SplitType.VAL)
