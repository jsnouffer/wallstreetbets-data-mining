import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = [
    "INFJ",
    "ENTP",
    "INTP",
    "INTJ",
    "ENTJ",
    "ENFJ",
    "INFP",
    "ENFP",
    "ISFP",
    "ISTP",
    "ISFJ",
    "ISTJ",
    "ESTP",
    "ESFP",
    "ESTJ",
    "ESFJ",
]
unique_type_list = [x.lower() for x in unique_type_list]

cntizer = CountVectorizer(analyzer="word", max_features=1000, max_df=0.7, min_df=0.1)
tfizer = TfidfTransformer()

def translate_personality(personality):
    b_Pers = {"I": 0, "E": 1, "N": 0, "S": 1, "F": 0, "T": 1, "J": 0, "P": 1}
    return [b_Pers[l] for l in personality]


def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []

    for row in data.iterrows():

        # Remove and clean comments
        posts = row[1].posts

        # Remove url links
        temp = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            posts,
        )

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(" +", " ", temp).lower()

        # Remove multiple letter repeating words
        temp = re.sub(r"([a-z])\1{2,}[\s|\w]*", "", temp)

        lemmatiser = WordNetLemmatizer()
        # Remove stop words
        if remove_stop_words:
            temp = " ".join(
                [
                    lemmatiser.lemmatize(w)
                    for w in temp.split(" ")
                    if w not in useless_words
                ]
            )
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(" ")])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        # transform mbti to binary vector
        type_labelized = translate_personality(
            row[1].type
        )  # or use lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        # the cleaned data temp is passed here
        list_posts.append(temp)

    # returns the result
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


def transform(text: str, remove_stop_words=True, remove_mbti_profiles=True):
    text = re.sub(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        text,
    )

    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(" +", " ", text).lower()
    text = re.sub(r"([a-z])\1{2,}[\s|\w]*", "", text)

    lemmatiser = WordNetLemmatizer()
    if remove_stop_words:
        text = " ".join(
            [lemmatiser.lemmatize(w) for w in text.split(" ") if w not in useless_words]
        )
    else:
        text = " ".join([lemmatiser.lemmatize(w) for w in text.split(" ")])

    if remove_mbti_profiles:
        for t in unique_type_list:
            text = text.replace(t, "")

    # vec = cntizer.transform([text])

    # return tfizer.transform(vec).toarray()
    return text
