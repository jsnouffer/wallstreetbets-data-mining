import numpy as np
from .config import ConfigContainer, ConfigService
from dependency_injector.wiring import Provide
from keras import models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras import backend as K

MAX_VOCAB = 500


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight("balanced", [0.0, 1.0], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true))
            * (weights[:, 1] ** (y_true))
            * K.binary_crossentropy(y_true, y_pred),
            axis=-1,
        )

    return weighted_loss


def load_model(config: ConfigService = Provide[ConfigContainer.config_svc].provider()):
    weighted_model = models.load_model(
        config.property("modelLocation") + "weighted/", compile=False
    )
    weighted_model.compile()
    unweighted_model = models.load_model(
        config.property("modelLocation") + "unweighted/", compile=False
    )
    unweighted_model.compile()
    return [weighted_model, unweighted_model]


def classify(
    sample,
    models,
    labels,
):
    predictions = {}
    try:
        pred0 = models[0].predict([sample])
        pred1 = models[1].predict([sample])
        prediction = (pred0 + pred1) / 2.0
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


def get_model(sequences, targets, loss_function):
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
        loss=loss_function,
        optimizer=Adam(1e-4),
        metrics=[CategoricalCrossentropy()],
    )

    return model


def train_model(
    sequences,
    targets,
    model,
):

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5)],
    )

    # pred = model.predict(X_test)

    # THRESH = 0.5
    # for i in range(len(labels)):
    #     y_true = y_test[:, i]
    #     y_pred = (pred[:, i] > THRESH).astype(int)
    #     print(f"======={labels[i]}")
    #     print(classification_report(y_true, y_pred))

    # model.save(config.property("modelLocation"))


def evaluate_models(sequences, targets, labels, models):
    _, X_test, _, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    pred0 = models[0].predict(X_test)
    pred1 = models[1].predict(X_test)

    pred = np.mean(np.array([pred0, pred1]), axis=0)
    THRESH = 0.5
    for i in range(len(labels)):
        y_true = y_test[:, i]
        y_pred = (pred[:, i] > THRESH).astype(int)
        print(labels[i])
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))


def train(
    sequences,
    targets,
    labels,
    config: ConfigService = Provide[ConfigContainer.config_svc].provider(),
):
    class_weights = calculating_class_weights(targets)
    weighted_model = get_model(sequences, targets, get_weighted_loss(class_weights))
    unweighted_model = get_model(
        sequences, targets, BinaryCrossentropy(from_logits=False)
    )

    train_model(sequences, targets, weighted_model)
    train_model(sequences, targets, unweighted_model)

    weighted_model.save(config.property("modelLocation") + "weighted/")
    unweighted_model.save(config.property("modelLocation") + "unweighted/")
