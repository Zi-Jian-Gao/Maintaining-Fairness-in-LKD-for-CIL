import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    
    parser.add_argument('--config', type=str, default='./exps/lwf.json',
                        help='config file.')
    
    parser.add_argument('--device', type=str, default=["2","3"],
                        help='used gpu ids.')
    
    parser.add_argument('--increment', type=int, default=5,
                        help='the increment class number.')
    
    parser.add_argument('--init_cls', type=int, default=50,
                        help='the initial class number.')
    
    parser.add_argument('--method', type=str, default="normal",
                        help='is optinal, seen in models/ .py.')
    
    parser.add_argument('--dataset', type=str, default="cifar100",
                        help='used dataset.')
    
    parser.add_argument('--loadpre', type=int, default=0,
                        help='if 1 ,used for load checkpoint of 1st phase.')
    
    parser.add_argument('--lambd',type = float,default=1,
                        help='KD loss weight.')

    parser.add_argument('--alpha',type = float,default=1 ,
                        help='is helpful for DER.')

    parser.add_argument('--path',type = str,default="temp.pth" ,
                        help='is helpful for DER')

    return parser


if __name__ == '__main__':
    main()
