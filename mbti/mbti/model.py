import pickle

from .bert import BertTransformer
from .config import ConfigContainer, ConfigService
from dependency_injector.wiring import Provide
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import BertModel, BertTokenizer
from xgboost import XGBClassifier

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


def load_models(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    models = {}
    for p_type in personality_type:
        url: str = config.property("modelFolder") + p_type + ".p"
        print("loading " + url)
        with open(url, "rb") as file:
            model = pickle.load(file)
            models[p_type] = model
    return models

def classify(sample, models, config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    predictions = {}
    type_array = ['','','','']
    for p_type in models.keys():
        prediction = models[p_type].predict([sample])[0]
        if p_type == "IE":
            predictions["energy"] = "introversion" if prediction == 0 else "extraversion"
            type_array[0] = "I" if prediction == 0 else "E"
        elif p_type == "NS":
            predictions["information"] = "intuition" if prediction == 0 else "sensing"
            type_array[1] = "N" if prediction == 0 else "S"
        elif p_type == "FT":
            predictions["decisions"] = "feeling" if prediction == 0 else "thinking"
            type_array[2] = "F" if prediction == 0 else "T"
        elif p_type == "JP":
            predictions["lifestyle"] = "judging " if prediction == 0 else "perceiving"
            type_array[3] = "J" if prediction == 0 else "P"

    predictions["type"] = type_array[0] + type_array[1] + type_array[2] + type_array[3]
    return predictions

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
            model, open(config.property("modelFolder") + personality_type[l] + ".p", "wb")
        )
