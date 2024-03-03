import pathlib
import shutil
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

dataset_lib = pathlib.Path("dataset/train")
image_count = len(list(dataset_lib.glob("*/*.jpg")))

batch_size = 32
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_lib,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_lib,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights('D:/Projects/Python/science-project/model/architecture')

loss, acc = model.evaluate(train_ds, verbose=2)

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
        image = tf.keras.utils.load_img(dirInput + "/" + item, target_size=(img_height, img_width))
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)

        predictions = model.predict(image_array)
        score = tf.nn.softmax(predictions[0])

        if (float(np.max(score) * 100) >= 30):
            shutil.move(dirInput + "/" + item, sortedDirInput + "/" + class_names[np.argmax(score)] + "/" + item)
        else:
            shutil.move(dirInput + "/" + item, sortedDirInput + "/other/" + item)
        
        print(class_names[np.argmax(score)], np.max(score) * 100, item)
else:
    print("Directory dont found!")