# %% [code]
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Function to preprocess the tweets data
def preprocess_tweet_data(data, name):
    # Proprocessing the data
    data[name] = data[name].str.lower()
    # Code to remove the Hashtags from the text
    data[name] = data[name].apply(lambda x: re.sub(r"\B#\S+", " ", x))
    # Code to remove the links from the text
    data[name] = data[name].apply(lambda x: re.sub(r"http\S+", " ", x))
    # Code to remove the Special characters from the text
    data[name] = data[name].apply(lambda x: " ".join(re.findall(r"\w+", x)))
    # Remove the twitter handlers
    data[name] = data[name].apply(lambda x: re.sub("@[^\s]+", " ", x))
    # Code to substitute the multiple spaces with single spaces
    data[name] = data[name].apply(lambda x: re.sub(r"\s+", " ", x, flags=re.I))
    # Code to remove all the single characters in the text
    data[name] = data[name].apply(lambda x: re.sub(r"\s+[a-zA-Z]\s+", " ", x))
    return data


# This function is to remove stopwords from a particular column and to tokenize it
def rem_stopwords_tokenize(data, name):
    def getting(sen):
        example_sent = sen

        stop_words = set(stopwords.words("english"))

        word_tokens = word_tokenize(example_sent)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w.lower())
        return filtered_sentence

    x = []
    for i in data[name].values:
        x.append(getting(i))
    data[name] = x
    return data


# Making a function to lemmatize all the words
lemmatizer = WordNetLemmatizer()


def lemmatize_all(data, name):
    arr = data[name]
    a = []
    for i in arr:
        b = []
        for j in i:
            x = lemmatizer.lemmatize(j, pos="a")
            x = lemmatizer.lemmatize(x, pos="v")
            x = lemmatizer.lemmatize(x)
            b.append(x)
        a.append(b)
    data[name] = a
    return data


def lemmatize_single_word(word):
    x = lemmatizer.lemmatize(word, pos="a")
    x = lemmatizer.lemmatize(x, pos="v")
    x = lemmatizer.lemmatize(x)
    return x


# Function to make it back into a sentence
def make_sentences(data, name):
    data[name] = data[name].apply(lambda x: " ".join([i + " " for i in x]))
    # Removing double spaces if created
    data[name] = data[name].apply(lambda x: re.sub(r"\s+", " ", x, flags=re.I))
    return data
