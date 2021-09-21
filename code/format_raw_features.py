"""
Accepts a feature input file and formats it in TSV form.

PAK  2014-01-01 <feat 1> <feat 2> ... <feat n>

If --fill-missing-dates is selected then output file has all of the dates
in range 2014-01-01 to 2019-12-31. The "empty" dates are filled with zeros.
"""
import os
import sys
import argparse
import logging
import pickle
import re

from pandas import date_range

logging.basicConfig(level=logging.DEBUG)


def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-file", type=str, required=True)
        parser.add_argument("--output-file", type=str, required=True)
        parser.add_argument("--fill-missing-dates", action="store_true")
        parser.add_argument("--fill-value", default=0)
        parser.add_argument("--header", action="store_true",
                            help="Uses first line of file as feature names")
        parser.add_argument("--delimiter", default="\t")
        parser.add_argument("--index", action="store_true",
                            help="Whether there is an index at the start of each row. "
                                 "Will not be included in output file.")
        parser.add_argument("--print-missing-stats", action="store_true")
        return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load country dict
    # Renaming 2 letter country code with 3 letter code
    with open(f"{os.getenv('MINERVA_HOME')}/data/country_code_to_UN_code_dict.pkl", "rb") as f:
        country_codes = pickle.load(f)
        all_countries = set(country_codes.values())

    # Get all dates
    filename = args.input_file.split("/")[-1]
    if "train" in filename:
        start_date = "1/1/2014"
        end_date = "12/31/2017"
    elif "test" in filename:
        start_date = "1/1/2018"
        end_date = "12/31/2019"
    else:
        start_date = "1/1/2014"
        end_date = "12/31/2019"
    all_dates = set([d.strftime("%Y-%m-%d") for d in date_range(start_date, end_date)])

    check = len(all_dates) * len(all_countries)

    DOCUMENT_RE = re.compile(r"(\d{4}_\d{2}_\d{2})_([A-Z]{2})")
    
    formatted_output = []
    feature_names = None
    current_dates = set()
    # Load input file
    with open(args.input_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if args.header and i == 0:
                feature_names = [s.strip() for s in line.split(args.delimiter)[1:]]
                continue
            # Check for in-line comments
            if line.startswith("#"):
                continue
            line = [s.strip() for s in line.split(args.delimiter)]

            # Have: 0 file:/export/fs04/a13/alexandra/minerva/lda_input_en/2014_02_12_MA.gz features ...
            # Want: MA 2014-02-12 features
            # Remove index if present
            if args.index:
                line = line[1:]
            try:
                date, country = DOCUMENT_RE.findall(line[0])[0]
            except IndexError as err:
                logging.warning(f"line in {args.input_file} does not match {DOCUMENT_RE}: {line}")
                continue

            date = re.sub("_", "-", date)
            country = country_codes.get(country)

            row = [country, date]
            row.extend(line[1:])  # feature vector
            row = args.delimiter.join(row)

            formatted_output.append(row)
            current_dates.add(f"{date}-{country}")
            
    # Open output file
    logging.info(f"There are {len(current_dates)} dates in the dataset ({len(current_dates)/check:.2%} of all dates)")
    feature_length = len(formatted_output[0].split(args.delimiter)) - 2

    if args.fill_missing_dates:
        # Write out and fill in missing dates
        counter = 0
        with open(args.output_file, "w+") as f:
            # Write feature vectors
            f.write("\n".join(formatted_output) + "\n")
            counter += len(current_dates)
            # Add missing dates to output file
            empty_vector = args.delimiter.join([f"{args.fill_value}" for i in range(feature_length)])
            for country in all_countries:
                missing_counter = 0
                for date in all_dates:
                    if f"{date}-{country}" not in current_dates:
                        f.write(f"{country}\t{date}\t{empty_vector}\n")
                        missing_counter += 1
                        counter += 1
            if args.print_missing_stats:
                print(f"{country},{missing_counter},{missing_counter / len(all_dates):.2%}")
            if counter != check:
                logging.error(f"Wrong number of values written. Should be {check} but is {counter}")
                sys.exit(1)
    else:
        # Write other file with formatted data without filling in missing dates
        with open(args.output_file, "w+") as f:
            # Write column names
            if feature_names:
                header_row = f"COUNTRY{args.delimiter}DATE{args.delimiter}"\
                             + args.delimiter.join(feature_names)
            else:
                header_row = f"COUNTRY{args.delimiter}DATE{args.delimiter}"\
                             + args.delimiter.join([f"feat_{i}" for i in range(feature_length)])
            f.write(header_row + "\n")
            # Write feature vectors
            f.write("\n".join(formatted_output) + "\n")

