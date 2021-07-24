from .config import ConfigContainer, ConfigService
from dependency_injector.wiring import Provide
from keras import models
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.errors_impl import InvalidArgumentError


MAX_VOCAB = 500


def load_model(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    return models.load_model(config.property("modelLocation"))

def classify(sample, model, labels, config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    predictions = {}
    try:
        prediction = model.predict([sample])
    except InvalidArgumentError as e:
        print(e.message)
        return predictions

    for l in range(len(labels)):
        predictions[labels[l]] = True if prediction[0][l] > 0.5 else False
    
    return predictions

def get_encoder(sequences):
    encoder = TextVectorization(
        max_tokens=MAX_VOCAB, standardize="lower_and_strip_punctuation"
    )

    encoder.adapt(sequences)
    return encoder


def get_model(sequences, targets):
    encoder = get_encoder(sequences)

    model = Sequential(
        [
            encoder,
            Embedding(
                input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True
            ),
            LSTM(64),
            Dense(64, activation="relu"),
            Dense(targets.shape[1], activation="sigmoid"),
        ]
    )

    model.compile(
        loss=BinaryCrossentropy(from_logits=False),
        optimizer=Adam(1e-4),
        metrics=[CategoricalCrossentropy()],
    )

    return model


def train(sequences, targets, labels, config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    model = get_model(sequences, targets)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.1, random_state=42
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5)],
    )

    pred = model.predict(X_test)

    THRESH = 0.5
    for i in range(len(labels)):
        y_true = y_test[:, i]
        y_pred = (pred[:, i] > THRESH).astype(int)
        print(f"======={labels[i]}")
        print(classification_report(y_true, y_pred))

    model.save(config.property("modelLocation"))

