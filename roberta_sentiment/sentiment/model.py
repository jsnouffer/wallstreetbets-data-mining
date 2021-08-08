import csv
import numpy as np
import urllib.request

from .config import ConfigContainer, ConfigService
from .text_wrangle import pre_process
from dependency_injector.wiring import Provide
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TENSOR_SIZE = 512

class PretrainedRobertaModel:
    config: ConfigService = Provide[ConfigContainer.config_svc].provider()

    def __init__(self, task: str):
        self.task = task
        self.labels = self.__get_labels()
        self.model_name = self.config.property("transformersModel").replace(
            "{task}", self.task
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def __get_labels(self):
        mapping_link = self.config.property("transformersModelLabelMapping").replace(
            "{task}", self.task
        )
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        return [row[1] for row in csvreader if len(row) > 1]


    def classify(self, text) -> dict:
        predictions = {}

        encoded_input = self.tokenizer(text, return_tensors="pt")

        if encoded_input['input_ids'].shape[1] > MAX_TENSOR_SIZE:
            encoded_input = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TENSOR_SIZE)
            predictions['input_truncated'] = True

        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]

        predictions["class"] = self.labels[ranking[0]]
        for i in range(scores.shape[0]):
            label = self.labels[ranking[i]]
            score = scores[ranking[i]]
            predictions[label] = score.item()
        return predictions


def load_models():
    tasks = ["emotion", "sentiment"]
    models = {}

    for task in tasks:
        models[task] = PretrainedRobertaModel(task)

    return models


def classify(text: str, models: dict) -> dict:
    text = pre_process(text)

    predictions = {}
    for task in models.keys():
        predictions[task] = models[task].classify(text)
    return predictions
