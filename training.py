import cv2
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from settings.logging_config import logging

logger = logging.getLogger(__name__)

IMG_SIZE = 224


def load_data():
    x = list()
    y = list()
    doc_name = ["error", "normal"]
    for name in doc_name:
        for fileNames in os.listdir(f"./data/{name}"):
            img = cv2.imread(f'./data/{name}/{fileNames}')[..., ::-1]
            resized_arr = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x.append(resized_arr)
            if name == "error":
                y.append(1)
            else:
                y.append(0)
    return (x, y)


def image_data_generator_model():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    return (datagen)


def data_set_split(X, Y, test_size=0.3):
    datagen = image_data_generator_model()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=1)

    ###圖片大小256 x 256
    X_train = np.array(X_train) / 255
    X_test = np.array(X_test) / 255

    X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)

    X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(y_test)
    datagen.fit(X_train)

    return (X_train, X_test, y_train, y_test)


def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(224, 224, 3)))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(46, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    return model


def main(model_name: str, eps: int):
    x, y = load_data()
    logger.debug(f"len of data: {len(x)}")

    X_train, X_test, y_train, y_test = data_set_split(x, y, test_size=0.3)
    logger.debug(f"len of training: {len(X_train)}")
    logger.debug(f"len of testing: {len(X_test)}")

    ##train
    model = create_cnn_model()
    logger.debug(f"create_model: {model}")

    history = model.fit(X_train, y_train, epochs=eps)

    model.save(f"./models/{model_name}")

    model_history = history.history

    loss = model_history.get("loss")
    accuracy = model_history.get("accuracy")

    for i in range(0, 10):
        logger.debug(
            '*' * 10 + f"eps {i}" + '*' * 10 + "\n" + f'loss: {str(loss[i]):.6} accuracy: {str(accuracy[i]):.6}')

    return (history)


if __name__ == "__main__":
    x, y = load_data()
    X_train, X_test, y_train, y_test = data_set_split(x, y)

    ##train
    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=20)

    ##predit
    pred = model.predict(X_test)

    df_pred = pd.DataFrame(pred, columns=["predict_true", "predict_error"])
    df_pred["true"] = y_test

    model.save("./models/PLine-CNN-Model.v1.0.0.h5")

    #############test################
    model = load_model("./models/PLine-CNN-Model.v1.0.0.h5")
    pred = model.predict(X_test)
    df_pred = pd.DataFrame(pred, columns=["predict_true", "predict_error"])

    df_pred["true"] = y_test

    df_pred.to_csv("result.csv", encoding='utf-8-sig', index=False)
