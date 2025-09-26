import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--table", type=str, required=True)


def parse_ingestion_args():
    _parser.add_argument("--repo", type=str, required=True)
    _parser.add_argument("--branch", type=str, required=True)
    return _parser.parse_args()


def parse_inference_args():
    return _parser.parse_args()
