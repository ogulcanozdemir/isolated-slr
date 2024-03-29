from utils.experiment import Experiment
from utils.parameter_parser import parse_arguments


if __name__ == '__main__':
    params = parse_arguments()

    experiment = Experiment(opts=params)
    experiment.test_model()
