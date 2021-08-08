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
parser.add_argument(
    "--evaluate-model",
    metavar="evaluate_model",
    nargs="?",
    const=True,
    default=False,
)
args = parser.parse_args()


def generate_model(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    df = pd.read_csv(config.property("trainingData"))
    # df = df.head(500)
    logger.info(f"Input shape = {df.shape}")
    sequences, targets, labels = pre_process(df)
    train(sequences, targets, labels)

def evaluate_model(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    df = pd.read_csv(config.property("trainingData"))
    sequences, targets, labels = pre_process(df)
    models = load_model()
    evaluate_models(sequences, targets, labels, models)

def main(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    collection = mongo_connect().submission_records
    models = load_model()

    df = pd.read_csv(config.property("trainingData"), nrows=0)
    labels = get_labels(df)

    print("Started classification")
    for doc in collection.find(
        {
            "toxicity_ensemble": {"$exists": False},
        }
    ):

        results = {}
        if doc["title"]:
            results["title"] = classify(doc["title"], models, labels)

        if (
            doc["selftext"]
            and not doc["is_removed_by_author"]
            and not doc["is_removed_by_moderator"]
        ):
            results["text"] = classify(doc["selftext"], models, labels)

        collection.update_one(
            {"id": doc["id"]},
            {"$set": {"toxicity_ensemble": results}},
        )


if __name__ == "__main__":
    if args.generate_model:
        reply = str(input("Confirm to generate new model files (Yes): ")).strip()
        if reply == "Yes":
            generate_model()
    elif args.evaluate_model:
        evaluate_model()
    else:
        main()
