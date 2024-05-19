import shutil
import os
import PIL
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.activations import relu, linear, softmax
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers


dataset_lib = pathlib.Path("D:/Projects/Python/science-project/dataset")
image_count = len(list(dataset_lib.glob("*/*.jpg")))
print(f"Images: {image_count}")

batch_size = 32
img_width = 28
img_height = 28
img_channel = 1 # RGB - 3, color grayscale(black and white) - 1

full_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_lib,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    color_mode='grayscale'
)

x=[]
y=[]

for images, labels in full_ds:
    x.extend(images.numpy())
    y.extend(labels.numpy())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # 20% images for testing, other for training

# Convert x,y to array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# validation - for predict during training, test - for predict after training
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=42) # 10% for validation

# Convert from 0-255, to 0-1
x_train_normalized = x_train / 255
x_test_normalized = x_test / 255
x_val_normalized = x_val / 255

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("x_val: ", x_val.shape)

class_names = full_ds.class_names
num_classes = len(class_names)
print(f"Class names: {class_names}")

model_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(model_augmentation)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=sparse_categorical_crossentropy,
    metrics=['accuracy'], 
)

model.load_weights('D:/Projects/Python/science-project/model/architecture')

#"Resored model, accuracy: {:5.2f}%".format(100 * acc))

dirInput = str(input("Enter directory: "))
sortedDirInput = str(input("Enter directory for sorted images: "))

foundDir = pathlib.Path(dirInput)
foundSortdeDir = pathlib.Path(sortedDirInput)

print()

if foundDir.is_dir() == True and foundSortdeDir.is_dir() == True:
    imageExtensions = ['.jpg', '.jpeg', 'png']
    imageNames = []
    directoryNames = ['other']
    directoryNames = directoryNames + class_names

    for item in directoryNames:
        os.mkdir(sortedDirInput + "/" + item)

    for item in foundDir.iterdir():
        for imageExtensionsItem in imageExtensions:
            if item.name.endswith(imageExtensionsItem):
                imageNames.append(item.name)
    
    for item in imageNames:
        image = tf.keras.utils.load_img(dirInput + "/" + item, target_size=(img_height, img_width), color_mode='grayscale')
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)  # Corrected to expand along axis 0 for batch dimension

        predictions = model.predict(image_array)
        score = tf.nn.softmax(predictions[0])

        if (float(np.max(score) * 100) >= 30):
            shutil.move(dirInput + "/" + item, sortedDirInput + "/" + class_names[np.argmax(score)] + "/" + item)
        else:
            shutil.move(dirInput + "/" + item, sortedDirInput + "/other/" + item)
        
        print(class_names[np.argmax(score)], np.max(score) * 100, item)
else:
    print("Directory dont found!")
