from tensorflow import keras
from preprocessing import train_data, train_labels, test_data, test_labels
import matplotlib.pyplot as plt

filters = 64
model = keras.Sequential([
    keras.layers.Conv2D(filters, (3, 3), input_shape=train_data.shape[1:], activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters * 3, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters * 3, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(filters * 4, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=8, validation_split=0.20)


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
