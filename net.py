from tensorflow import keras
from tensorflow.keras import layers, Sequential
from src.preprocessing import train_data, test_data, train_labels, test_labels
import matplotlib.pyplot as plt

# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip("horizontal"),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#     ]
# )


def baseline_model(input_shape):
    model = Sequential()
    model.add(keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model


def make_model(input_shape, filters=32):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(filters, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters*2, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    n = filters * 4
    for size in [n, n*2, n*4, n*6]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(filters*32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)
    
model = make_model(input_shape=train_data.shape[1:], filters=64)
# model = baseline_model(train_data.shape[1:])

model.compile(optimizer=keras.optimizers.Adam(lr=1e-7),  # lr should be at least 1e-6
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=100, validation_split=0.25)

# model.summary()
def graphs(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()
graphs(history)

results = model.evaluate(test_data, test_labels)
print('test loss, test acc:', results)

model.save('model.h5', include_optimizer=False)
print("Saved model to disk")
