import argparse
import logging
import pandas as pd

from .model import *
from .mongo import mongo_connect
from .text_wrangle import *

logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--generate-model",
    metavar="generate_model",
    nargs="?",
    const=True,
    default=False,
)
args = parser.parse_args()


def generate_model():
    df = pd.read_csv("/home/jason/mbti_model/mbti_1.csv")
    logger.info(f"Input shape = {df.shape}")
    list_posts, list_personality = pre_process_text(
        df, remove_stop_words=True, remove_mbti_profiles=True
    )
    train(list_posts, list_personality)


def main():
    collection = mongo_connect().submission_records
    models = load_models()

    for doc in collection.find(
        {
            "is_self_post": True,
            "is_removed_by_author": False,
            "is_removed_by_moderator": False,
            "selftext": {"$ne": ""},
            "mbti": {"$exists": False},
        }
    ):

        text = transform(doc["selftext"])
        classification = classify(text, models)

        collection.update_one(
            {"id": doc["id"]},
            {"$set": {"mbti": classification}},
        )


if __name__ == "__main__":
    if args.generate_model:
        reply = str(input("Confirm to generate new model files (Yes): ")).strip()
        if reply == "Yes":
            generate_model()
    else:
        main()
