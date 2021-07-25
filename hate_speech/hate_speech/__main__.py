import logging

from .config import ConfigContainer, ConfigService
from .model import *
from .mongo import mongo_connect
from dependency_injector.wiring import Provide

logger = logging.getLogger("main")

def main(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    collection = mongo_connect().submission_records

    model = PretrainedBERTModel()

    print("Started classification")
    for doc in collection.find(
        {
            "hate-speech": {"$exists": False},
        }
    ):

        results = {}
        if doc["title"]:
            results["title"] = classify(doc["title"], model)

        if (
            doc["selftext"]
            and not doc["is_removed_by_author"]
            and not doc["is_removed_by_moderator"]
        ):
            results["text"] = classify(doc["selftext"], model)

        collection.update_one(
            {"id": doc["id"]},
            {"$set": {"hate-speech": results}},
        )


if __name__ == "__main__":
    main()
