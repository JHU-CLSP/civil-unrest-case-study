"""
Features that depend on the frequency of tokens in Tweets.

Can also be used to keyword counts if --valid-vocab file is passed.

Also supports only count civil unrest related tweets tagged by a filtration model. The filtration model tags tweets
with flag 'civil-unrest-related', and if --civil-unrest-only on, the features will only include counts from tweets whose
'civil-unrest-related' is True.

Author: Alexandra DeLucia, Jack Zhang
"""

# Standard
import os
import argparse
import logging
import gzip
import zlib
import pickle
from collections import Counter

# Third-party
import jsonlines as jl
import regex
from sklearn.feature_extraction.text import CountVectorizer

# Custom packages
from littlebird import TweetTokenizer, TweetReader

# Set up logging
logging.basicConfig(level=logging.INFO)


class TweetTokenCountAnalyzer:
    def __init__(self, token_pattern, language, stopwords_file=None):
        self.tokenizer = TweetTokenizer(token_pattern=token_pattern, language=language)

        if stopwords_file:
            with open(stopwords_file) as f:
                self.stopwords = [i.strip() for i in f.readlines()]
        else:
            self.stopwords = None

    def get_token_counts_from_file(self, input_file, scale_counts=False,
                                   sample_size=-1, valid_vocab=None, include_bigrams=False,
                                   min_count=5, civil_unrest_only=False):
        # Get cleaned text from file
        if civil_unrest_only:
            tweet_content = []
            reader = TweetReader(input_file)
            for t in reader.read_tweets():
                if t.get("civil_unrest_related", False):
                    tweet_content.append(self.tokenizer.get_tokenized_tweet_text(t))
        else:
            tweet_content = self.tokenizer.tokenize_tweet_file(input_file, sample_size=sample_size)
        logging.debug(tweet_content)
        num_tweets = len(tweet_content)

        # Get token counts
        try:
            tokens, counts, vectorizer = self.get_token_counts(tweet_content, include_bigrams=include_bigrams,
                                                   valid_vocab=valid_vocab, min_count=min_count)
        except ValueError as err:
            logging.warning(f"Issue counting tokens in {input_file}:\n{err}")
            return [], [], []

        # Scale the token counts by the number of Tweets
        if scale_counts:
            counts = counts / num_tweets
        return tokens, counts, vectorizer

    def get_token_counts(self, tweet_tokens, min_count, valid_vocab, include_bigrams):
        if include_bigrams:
            ngram_range = (1, 2)
        else:
            ngram_range = (1, 1)

        # Already tokenized, so just use whitespace as the tokenizer
        vectorizer = CountVectorizer(
            input="content",
            tokenizer=str.split,
            ngram_range=ngram_range,
            min_df=min_count,
            vocabulary=valid_vocab,
            stop_words=self.stopwords,
        )

        # Get token counts per tweet
        X = vectorizer.fit_transform(tweet_tokens).toarray()
        # Sum for total counts for this document
        X = X.sum(axis=0)
        return vectorizer.get_feature_names(), X, vectorizer


### Script methods ###
def aggregate_counts(output_dir):
    """Aggregates token counts saved in RAW format"""

    filenames = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".raw")]
    token_counts = Counter()
    for i, file in enumerate(filenames):
        if i % 100 == 0:
            logging.info(f"On file {i}")
        # Get tokens and their counts in file
        with open(file, "r") as f:
            # Update counts for each token
            try:
                for line in f.readlines():
                    try:
                        t, c = [str(i).strip() for i in line.split("\t")]
                        token_counts[t] += int(c)
                    except ValueError as err:
                        logging.error(f"Error parsing line {line} from file {file}: {err}")
                        continue
            except UnicodeDecodeError as err:
                logging.error(f"Error in file {file}. Skipping.\n{err}")
                continue

    return token_counts


def write_tsv(output_file, tokens, counts):
    with open(output_file, "w+") as f:
        header = "\t".join(tokens)
        f.write(f"#filename\t{header}\n")
        features = "\t".join(map(str, counts))
        f.write(f"{filename}\t{features}\n")


def write_raw_counts(output_file, tokens, counts):
    with open(output_file, "w+") as f:
        for t, c in zip(tokens, counts):
            f.write(f"{t}\t{c}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str, nargs="+", help="List of GZIP'd Tweet files")
    parser.add_argument("--output-dir")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-format", choices=["raw", "tsv"], default="tsv")
    parser.add_argument("--no-overwrite", action="store_true", help="If output file already exists, do not replace it")
    parser.add_argument("--aggregate", action="store_true", help="Use this flag to aggregate all the token counts")

    # Feature settings
    parser.add_argument("--stopwords", type=str, help="File with newline-delimited stopwords")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--include-bigrams", action="store_true")
    parser.add_argument("--token-pattern", type=str, default="\p{L}[\p{L}\p{P}]+\p{L}",
                        help="Regex pattern for matching tokens in tweets")
    parser.add_argument("--scale", action="store_true", help="Scale keyword counts by the number of tweets")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Number of tweets to use for the keyword counts. Only for Tweet files.")
    parser.add_argument("--language", choices=["en", "ar"], default="en")
    parser.add_argument("--valid-vocab", type=str,
                        help="Limit vocabulary to specific words in the provided file (newline-deliminted) after the tokenization)")
    parser.add_argument("--civil-unrest-only", action="store_true",
                        help="Only count tweets whose 'civil_unrest_related' flag is set to true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # Token aggregation use case
    if args.aggregate:
        # Get all counts
        all_token_counts = aggregate_counts(args.output_dir)
        with open(os.path.join(args.output_dir, "all_token_counts.pkl"), "wb") as f:
            pickle.dump(all_token_counts, f, protocol=3)

        # Save top 10K tokens
        with open(os.path.join(args.output_dir, "top_10k_tokens.txt"), "w+") as f:
            output = "\n".join([token for token, count in all_token_counts.most_common(10000)])
            f.write(output)

        # Exit
        quit()

    # Token analysis use case
    analyzer = TweetTokenCountAnalyzer(args.token_pattern, args.language, args.stopwords)

    if args.valid_vocab is not None:
        # Load valid vocab
        with open(args.valid_vocab) as f:
            valid_vocab = set([i.strip() for i in f.readlines()])
    else:
        valid_vocab = None

    # Process Tweet file
    for i, input_file in enumerate(args.input_files):
        filename = input_file.split("/")[-1]
        if args.output_format == "tsv":
            output_file = os.path.join(args.output_dir, filename) + ".tsv"
        elif args.output_format == "raw":
            output_file = os.path.join(args.output_dir, filename) + ".raw"

        if args.no_overwrite and os.path.exists(output_file):
            logging.warning(f"Skipping: {output_file} already exists.")
            continue

        # Count the tokens in the file
        tokens, counts, vectorizer = analyzer.get_token_counts_from_file(
            input_file,
            valid_vocab=valid_vocab,
            sample_size=args.sample,
            scale_counts=args.scale,
            include_bigrams=args.include_bigrams,
            min_count=args.min_count,
            civil_unrest_only=args.civil_unrest_only
        )

        # Save output
        # TSV format is for final features
        # Raw is for aggregating the counts
        if args.output_format == "tsv":
            write_tsv(output_file, tokens, counts)
        elif args.output_format == "raw":
            write_raw_counts(output_file, tokens, counts)

        # Save vectorizer
        if i == 0:
            with open(f"{args.output_dir}/vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
