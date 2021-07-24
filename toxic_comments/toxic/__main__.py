import argparse
import logging
import pandas as pd

from .config import ConfigContainer, ConfigService
from .model import *
from .mongo import mongo_connect
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


def generate_model(
    config: ConfigService = Provide[ConfigContainer.config_svc].provider(),
):
    df = pd.read_csv(config.property("trainingData"))
    logger.info(f"Input shape = {df.shape}")
    sequences, targets, labels = pre_process(df)
    train(sequences, targets, labels)


def main(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    collection = mongo_connect().submission_records
    model = load_model()

    df = pd.read_csv(config.property("trainingData"), nrows=0)
    labels = get_labels(df)

    print("Started classification")
    for doc in collection.find(
        {
            "toxicity": {"$exists": False},
        }
    ):

        results = {}
        if doc["title"]:
            results["title"] = classify(doc["title"], model, labels)

        if (
            doc["selftext"]
            and not doc["is_removed_by_author"]
            and not doc["is_removed_by_moderator"]
        ):
            results["text"] = classify(doc["selftext"], model, labels)

        collection.update_one(
            {"id": doc["id"]},
            {"$set": {"toxicity": results}},
        )


if __name__ == "__main__":
    if args.generate_model:
        reply = str(input("Confirm to generate new model files (Yes): ")).strip()
        if reply == "Yes":
            generate_model()
    else:
        main()
