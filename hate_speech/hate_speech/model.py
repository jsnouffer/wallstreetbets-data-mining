from .config import ConfigContainer, ConfigService
from .text_wrangle import pre_process
from dependency_injector.wiring import Provide
from happytransformer import HappyTextClassification

class TruncatingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs['truncation'] = True
        return self.tokenizer.__call__(*args, **kwargs)

class PretrainedBERTModel:
    config: ConfigService = Provide[ConfigContainer.config_svc].provider()

    def __init__(self):
        self.model = HappyTextClassification(
            self.config.property("transformers.type"),
            self.config.property("transformers.model"),
            int(self.config.property("transformers.labels")),
        )
        self.model_w_truncation = HappyTextClassification(
            self.config.property("transformers.type"),
            self.config.property("transformers.model"),
            int(self.config.property("transformers.labels")),
        )
        self.model_w_truncation._pipeline.tokenizer = TruncatingTokenizer(self.model_w_truncation._pipeline.tokenizer)

    def classify(self, text):
        results = {}
        prediction = None
        try:
            prediction = self.model.classify_text(text)
        except RuntimeError:
            prediction = self.model_w_truncation.classify_text(text)
            results['input_truncated'] = True

        results['class'] = prediction.label
        results['score'] = prediction.score
        return results


def classify(text: str, model) -> dict:
    text = pre_process(text)

    return model.classify(text)