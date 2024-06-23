import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
import cv2

# from google.colab import drive
# drive.mount('/content/drive')

train_directory = '/content/drive/MyDrive/mat/train'

BATCH_SIZE = 32
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  # Điều chỉnh độ sáng ngẫu nhiên
    validation_split=0.3  # 30% of the data will be used for validation
)

# Load the training data
train_set= datagen.flow_from_directory(
    train_directory,
    target_size=(416, 416),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
)
val_set= datagen.flow_from_directory(
    train_directory,
    target_size=(416, 416),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
)
input_tensor = Input(shape=(416,416,3)) # dau vao
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
BatchNormalization()
x = Dense(2, activation='softmax')(x)

model = Model(inputs = input_tensor, outputs = x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
H = model.fit(train_set, epochs=200, validation_data=val_set, steps_per_epoch= train_set.samples // BATCH_SIZE, validation_steps= val_set.samples // BATCH_SIZE)

# Kiểm tra kết quả độ chính xác trên tập validation
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
from tensorflow.keras.models import load_model

# Tải mô hình từ file 'model.h5'
model = load_model('/content/drive/MyDrive/NhanDienMat.hdf5')
loss, accuracy = model.evaluate(val_set, steps=len(val_set), verbose=1)

