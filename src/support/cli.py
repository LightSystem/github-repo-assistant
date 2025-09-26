import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--branch", type=str, required=True)
    return parser.parse_args()
