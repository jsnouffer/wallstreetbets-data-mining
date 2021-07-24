import logging
from .config import ConfigContainer, ConfigService
from dependency_injector.wiring import Provide
from pymongo import MongoClient

logger = logging.getLogger("main")

def mongo_connect(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    client: MongoClient = MongoClient(
        host=config.property("mongoUrl"),
        serverSelectionTimeoutMS=5000,
    )

    try:
        client.admin.command("ismaster")
    except Exception:
        logger.error("Mongo server not available at " + config.property("mongoUrl"))
        raise

    db = client[config.property("mongoDB")]
    return db
