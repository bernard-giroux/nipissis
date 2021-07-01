from argparse import ArgumentParser

from figures import *
from catalog import catalog


parser = ArgumentParser()
parser.add_argument('--metadata', '-m', action='store_true')
parser.add_argument('--figure', '-f', type=str)
args = parser.parse_args()

if args.metadata:
    if args.figure:
        catalog.regenerate(args.figure)
    else:
        catalog.regenerate_all()

catalog.draw_all(show=False)
