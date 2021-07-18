import emoji
import logging

from .config import ConfigContainer, ConfigService
from csv import DictReader
from dependency_injector.wiring import Provide
from praw import Reddit
from pymongo import MongoClient

logger = logging.getLogger("main")


def mongo_connect(url: str):
    client: MongoClient = MongoClient(
        host=url,
        serverSelectionTimeoutMS=5000,
    )

    try:
        client.admin.command("ismaster")
    except Exception:
        logger.error("Mongo server not available at " + url)
        raise

    db = client["wallstreetbets"]
    return db


def reddit_connect(config_svc: ConfigService):
    logger.debug("connecting to Reddit")
    reddit = Reddit(
        client_id=config_svc.property("reddit.client-id"),
        client_secret=config_svc.property("reddit.client-secret"),
        user_agent="my user agent",
    )
    logger.info("connected to Reddit")
    return reddit

collection_name: str = "submission_records"
def main(config_svc: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    db = mongo_connect(config_svc.property("mongoUrl"))
    reddit = reddit_connect(config_svc)

    with open(
        "/home/jason/wallstreetbets/data/kaggle_012821_to_071621/reddit_wsb.csv", "r"
    ) as read_obj:
        for row in DictReader(read_obj):
            if bool(db[collection_name].find_one({"id": row["id"]})):
                continue

            submission = reddit.submission(id=row["id"])

            submission_record = {
                "id": submission.id,
                "permalink": submission.shortlink,
                "author": submission.author.name if submission.author else "",
                "author_deleted": submission.author == None,
                "created_utc": submission.created_utc,
                "title": submission.title,
                "selftext": submission.selftext,
                "url": submission.url,
                "is_self_post": submission.is_self,
                "num_comments": submission.num_comments,
                "is_nsfw": submission.over_18,
                "score": submission.score,
                "is_stickied": submission.stickied,
                "upvote_ratio": submission.upvote_ratio,
                "distinguished": submission.distinguished,
                "edited_utc": submission.edited if submission.edited != False else "",
                "is_edited": submission.edited != False,
                "is_original_content": submission.is_original_content,
                "link_flair_text": submission.link_flair_text,
                "is_locked": submission.locked,
                "is_removed_by_moderator": submission.selftext == "[removed]",
                "is_removed_by_author": submission.selftext == "[deleted]",
            }
            
            title_emoji = []
            for raw_emoji in emoji.distinct_emoji_lis(submission.title):
                title_emoji.append(emoji.demojize(raw_emoji))
            submission_record["title_emoji"] = title_emoji

            body_emoji = []
            for raw_emoji in emoji.distinct_emoji_lis(submission.selftext):
                body_emoji.append(emoji.demojize(raw_emoji))
            submission_record["body_emoji"] = body_emoji

            db[collection_name].update_one({'id': submission_record['id']}, {'$set': submission_record}, upsert=True)


if __name__ == "__main__":
    main()
