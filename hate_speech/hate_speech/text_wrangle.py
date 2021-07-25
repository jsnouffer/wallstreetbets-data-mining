import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()
useless_words = stopwords.words("english")


def pre_process(text: str):
    # Remove url links
    text = re.sub(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        text,
    )

    # Remove Non-words - keep only words
    text = re.sub("[^a-zA-Z]", " ", text)

    # Remove spaces > 1
    text = re.sub(" +", " ", text)

    # Remove stop words
    text = " ".join(
        [lemmatiser.lemmatize(w) for w in text.split(" ") if w not in useless_words]
    )

    return text
