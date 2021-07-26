import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--resampler',
    choices = ['nn', 'lw'],
    type=str,
    help='Type of resampler. NN or LW'
)

args, _ = parser.parse_known_args()
print(args.resampler)