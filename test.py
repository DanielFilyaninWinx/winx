import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image

from createMnist import X_test, model

json_file = open('mnist_model_two.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mnist_model_two.h5")
print("Loaded model from disk")

n_rec = 487
plt.imshow(X_test[n_rec].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

#Меняем размерность изображения и нормализуем его
x = X_test[n_rec]
x = np.expand_dims(x, axis=0)

#запускаем распоз
prediction = model.predict(x)
prediction

#Преобразуем результаты из формата one hot encoding
prediction = np.argmax(prediction[0])
print("Ответ:", prediction)

#Загружаем свою картинку
name='number.jpg'
img_path = input('doc//'+name)
img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
plt.imshow(img.convert('RGBA'))
plt.show()

# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

prediction = loaded_model.predict(x)
prediction
prediction = np.argmax(prediction)
print("ответ:", prediction)
