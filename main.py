import argparse
import yaml

from utils import *
from trainer import *


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument("--config_path",
                        type=str,
                        default='config.yaml',
                        help="Config path of trainer.")

    return parser.parse_args()


def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a G-UNet model.
    """
    args = parameter_parser()
    config = load_config(args.train_config_path)
    trainer = Trainer(config)
    print(f"Total parameters:{calculate_total_params(trainer.model)}")
    if config['load_model']:
        trainer.load()
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()