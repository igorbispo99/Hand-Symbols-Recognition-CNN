import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense, AveragePooling2D, Dropout, Conv2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

IMG_SHAPE = (45, 45, 1)
NUM_CLASSES = 82

def instantiate_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size = (3, 3) , activation='relu', input_shape=(45, 45, 1)))
    model.add(Conv2D(32, kernel_size = 2))
    model.add(Conv2D(64, kernel_size = 2))            

    model.add(Conv2D(64, kernel_size = 2 , activation='relu'))
    model.add(Conv2D(64, kernel_size = 2, activation='relu'))
    model.add(Conv2D(64, kernel_size = 2, activation='relu'))                

    model.add(Conv2D(64, kernel_size = 2 , activation='relu'))
    model.add(Conv2D(64, kernel_size = 2, activation='relu'))
    model.add(Conv2D(128, kernel_size = 2, activation='relu'))            
    model.add(AveragePooling2D(pool_size=2))

    model.add(Conv2D(128, kernel_size = 2 , activation='relu'))
    model.add(Conv2D(128, kernel_size = 2, activation='relu'))
    model.add(Conv2D(128, kernel_size = 2, activation='relu'))           

    model.add(Conv2D(128, kernel_size = 2 , activation='relu') )
    model.add(Conv2D(128, kernel_size = 2, activation='relu'))
    model.add(Conv2D(256, kernel_size = 2, activation='relu'))            
    model.add(AveragePooling2D(pool_size=2 ))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    print(model.summary())

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

checkpoint = ModelCheckpoint("cnn_mathsym_v2.h5", 
                    monitor="acc",
                    verbose=1,
                    mode="max",
                    period=1,
                    save_weights_only=False,
                    save_best_only=True)

datagen = keras.preprocessing.image.ImageDataGenerator(
                        rescale=1./255)

data = datagen.flow_from_directory("extracted_images/", 
                        batch_size=128,
                        color_mode='grayscale',
                        target_size=(45, 45),
                        class_mode='categorical')

model = instantiate_model()
model.fit_generator(data, steps_per_epoch = 375974 // 64, epochs=10, callbacks=[checkpoint])

