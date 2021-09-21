"""
Add the civil unrest filtration score from the BERTweet model to the tweets

Author: Alexandra DeLucia
"""
import os
import argparse
import torch
import sys
sys.path.append("/home/aadelucia/files/minerva/src/feature_engineering")
sys.path.append("/home/aadelucia/files/minerva/emnlp_wnut_2020")
from bertweet_model import LogisticRegression
from BERTweet_utils import Batcher, BERTweetWrapper
from littlebird import TweetReader, TweetWriter
from littlebird import BERTweetTokenizer as TweetNormalizer

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", required=True, nargs="+", help="Twitter files")
    parser.add_argument("--output-dir", required=True, help="Output directory for file results")
    parser.add_argument("--model-path", help="Path to filtration model")
    parser.add_argument("--BERTweet-model-path",
        default="/home/aadelucia/files/minerva/src/feature_engineering/BERTweet_base_transformers",
        help="Path to BERTweet_base_transformers folder")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set CPU/GPU device and claim it
    if args.cpu:
        device = "cpu"
        torch.device("cpu")
    else:
        device = "cuda"
        torch.device("cuda")
        torch.ones(1).to("cuda")

    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # Initialize BERTweetWrapper for tweet representations
    wrapper = BERTweetWrapper(args.BERTweet_model_path, device, debug=args.debug)
    tokenizer = TweetNormalizer()

    # Load filtration model
    model = torch.load(args.model_path, map_location=device)

    for file in args.input_files:
        output_file = os.path.join(args.output_dir, file.split("/")[-1])

        # Load tweets
        reader = TweetReader(file)
        tweets, tweet_text = [], []
        for t in reader.read_tweets():
            tweets.append(t)
            tweet_text.append(tokenizer.get_tokenized_tweet_text(t))

        if len(tweets) == 0:
            logging.warning(f"{file} has no tweets. Skipping.")
            continue

        # Get BERTweet representation
        logging.info(f"Collecting BERTweet feature representations")
        features = wrapper.get_BERTweet_representation(tweet_text, pretokenized=True)
        logging.info(f"Created {len(features):,} tweet representations")

        # Get filtration model prediction
        # Want to store raw probability and boolean for tweets
        # Use threshold of 0.5 (same as the model)
        prediction_probs = model.predict_proba(features)
        discuss_unrest = prediction_probs > 0.5

        for (tweet, prob, is_unrest) in zip(tweets, prediction_probs, discuss_unrest):
            tweet["civil_unrest_related"] = is_unrest.item()
            tweet["civil_unrest_score"] = prob.item()

        # Write out to file
        writer = TweetWriter(output_file)
        writer.write(tweets)

