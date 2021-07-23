import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import BertModel, BertTokenizer
from xgboost import XGBClassifier

from .bert import BertTransformer

USE_BERT_MODEL = False

personality_type = [
    "IE",
    "NS",
    "FT",
    "JP",
]

bert_dataset = "bert-base-uncased"


def tfidf():
    return Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer())])


def bert():
    tokenizer = BertTokenizer.from_pretrained(bert_dataset)
    bert_model = BertModel.from_pretrained(bert_dataset)
    return BertTransformer(tokenizer, bert_model)


def train(list_posts, list_personality):
    for l in range(len(personality_type)):

        Y = list_personality[:, l]

        X_train, X_test, y_train, y_test = train_test_split(
            list_posts, Y, test_size=0.2, random_state=42
        )

        classifier = XGBClassifier()

        if USE_BERT_MODEL:
            model = Pipeline(
                [
                    (
                        "union",
                        FeatureUnion(
                            transformer_list=[("bert", bert()), ("tf_idf", tfidf())]
                        ),
                    ),
                    ("classifier", classifier),
                ]
            )
        else:
            model = Pipeline(
                [
                    ("vectorizer", tfidf()),
                    ("classifier", classifier),
                ]
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

        pickle.dump(
            model, open("/home/jason/mbti_model/" + personality_type[l] + ".p", "wb")
        )
