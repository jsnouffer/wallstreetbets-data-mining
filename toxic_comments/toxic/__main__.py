import argparse
import logging
import pandas as pd

from .config import ConfigContainer, ConfigService
from .model import *
from .text_wrangle import *
from dependency_injector.wiring import Provide

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


def generate_model(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    df = pd.read_csv(config.property("trainingData"))
    logger.info(f"Input shape = {df.shape}")
    sequences, targets, labels = pre_process(df)
    train(sequences, targets, labels)


def main():
    # collection = mongo_connect().submission_records
    model = load_models()
    print(model)

    # for doc in collection.find(
    #     {
    #         "is_self_post": True,
    #         "is_removed_by_author": False,
    #         "is_removed_by_moderator": False,
    #         "selftext": {"$ne": ""},
    #         "mbti": {"$exists": False},
    #     }
    # ):

    #     text = transform(doc["selftext"])
    #     classification = classify(text, models)

    #     collection.update_one(
    #         {"id": doc["id"]},
    #         {"$set": {"mbti": classification}},
    #     )


if __name__ == "__main__":
    if args.generate_model:
        reply = str(input("Confirm to generate new model files (Yes): ")).strip()
        if reply == "Yes":
            generate_model()
    else:
        main()
