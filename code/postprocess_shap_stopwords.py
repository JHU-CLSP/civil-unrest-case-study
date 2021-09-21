from collections import defaultdict
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def main(args):

    stopwords = set()

    with open(args.input) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 1:
                continue
            stopwords.add(tokens[0])

    logging.info(f"Length of produced stopwords list = {len(stopwords)}")

    with open(args.output) as f:
        for word in list(stopwords):
            print(word, file=f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
