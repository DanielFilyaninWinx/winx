import numpy as np
import matplotlib.pyplot as plt
import random

from keras.datasets import mnist
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

#подгружаем набор данных при помощи встроенных методов keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()#х_train - данные для обучения y_train - правильные ответы

plt.rcParams['figure.figsize'] = (9, 9)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    n = random.randint(0, len(X_train))
    plt.imshow(X_train[n], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[n]))

plt.tight_layout()

#Преобразование размерности данных в наборе
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

#Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#делим интенсивность каждого пикселя изображения на 255
X_train /= 255
X_test /= 255

#Работа с правильными ответами
print(y_train[n])

#Преобразуем метки в формат one hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Правильный ответ в формате one hot encoding
print(y_train[n])

#Создаем нейронную сеть
#Создаем последовательную модель
model = Sequential()

#Добавляем уровни/слои сети
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

#Компилируем сеть
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

#Обучаем нейронную сеть
model.fit(X_train, Y_train, batch_size=200, epochs=20,  verbose=1)

#Сохраняем обученную нейронную сеть
#Генерируем описание модели в формате json
model_json = model.to_json()
print(model_json)

#Сохраняем файлы на локальный компьютер
#Записываем архитектуру сети в файл
with open("mnist_model_two.json", "w") as json_file:
    json_file.write(model_json)

#Записываем данные о весах сети в файл
model.save_weights("mnist_model_two.h5")
print("Saved model to disk")

#Оценка качества обучения
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))