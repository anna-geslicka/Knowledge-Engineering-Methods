from keras import layers, models
import plot as plt


def run(x_train, y_train, x_test, y_test, x_eval, y_eval):
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(33, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test))

    plt.plot(history)

    model.summary()
    test_loss, test_acc = model.evaluate(x_eval, y_eval)
    print("\nloss:", round(test_loss, 2), "\naccuracy:", test_acc)



