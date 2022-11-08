import cv2
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

X = []
Y = []
for f in os.listdir("data/error"):
    print(f)
doc_name = ["error", "normal"]
img_size = 224
for name in doc_name:
    for fileNames in os.listdir(f"./{name}"):
        img = cv2.imread(f'./{name}/{fileNames}')[..., ::-1]
        resized_arr = cv2.resize(img, (img_size, img_size))
        X.append(resized_arr)
        if name == "error":
            Y.append(1)
        else:
            Y.append(0)
###圖片大小256 x 256

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

X_train = np.array(X_train) / 255
X_test = np.array(X_test) / 255

X_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

X_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

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

datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(X_train, y_train, epochs=10)

pred = model.predict(X_test)

df_pred = pd.DataFrame(pred, columns=["predict_true", "predict_error"])
df_pred["true"] = y_test

model.save("./models/PLine_error_detect.v1.0.0.h5")


model = load_model("./models/PLine_error_detect.v1.0.0.h5")
pred = model.predict(X_test)
df_pred = pd.DataFrame(pred, columns=["predict_true", "predict_error"])
df_pred["true"] = y_test

df_pred.to_csv("result.csv",encoding='utf-8-sig',index = False)