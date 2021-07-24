import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


stop_words = set(stopwords.words("english"))
stop_words.update(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "may",
        "also",
        "across",
        "among",
        "beside",
        "however",
        "yet",
        "within",
    ]
)
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

stemmer = SnowballStemmer("english")


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def get_labels(df):
    df = df.drop(["id", "set", "toxicity"], axis=1)
    labels = list(df.columns)
    labels.remove("comment_text")
    return labels

def pre_process(df):
    labels = get_labels(df)

    df["comment_text"] = df["comment_text"].apply(removeStopWords)
    df["comment_text"] = df["comment_text"].apply(stemming)

    sequences = df["comment_text"].values
    targets = df[labels].values

    return sequences, targets, labels
