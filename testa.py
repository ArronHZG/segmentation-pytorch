import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Segmentation')
# net and datasets
parser.add_argument('--net', type=str,help="sdfsd",required=True)
parser.add_argument('--backbone', type=str)

args = parser.parse_args()

print(args)