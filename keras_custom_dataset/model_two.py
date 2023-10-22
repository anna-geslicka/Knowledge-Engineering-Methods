from keras import layers, models
import tensorflow as tf
import plot as plt

height = 15
width = 15


def run(x_train, y_train, x_test, y_test, x_eval, y_eval):
    model = models.Sequential()
    model.add(tf.keras.layers.Resizing(height, width, interpolation='bilinear', crop_to_aspect_ratio=False))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid', input_shape=(height, width, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(33, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=45, validation_data=(x_test, y_test))

    plt.plot(history)

    test_loss, test_acc = model.evaluate(x_eval, y_eval)
    model.summary()
    print("\nloss:", round(test_loss, 2), "\naccuracy:", test_acc)


