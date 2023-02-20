import argparse
import yaml

from trainer import *


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run DCUNet.")

    parser.add_argument("--train_config_path",
                        type=str,
                        default='./config/train_config.yaml',
                        help="Config path of trainer.")

    return parser.parse_args()


def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a G-UNet model.
    """
    args = parameter_parser()
    train_config = load_config(args.train_config_path)
    trainer = Trainer(train_config)
    if train_config['load_path']:
        trainer.load()
        trainer.train()
    else:
        trainer.train()
    trainer.pesq_score()


if __name__ == "__main__":
    main()