import matplotlib.pyplot as plt


def plot(history):
    loss = history.history['loss']
    acc = history.history['accuracy']
    v_loss = history.history['val_loss']
    v_acc = history.history['val_accuracy']
    plt.plot(acc, color='b', label='training accuracy')
    plt.plot(v_acc, color='g', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Wykres dokładności')
    plt.legend()
    plt.show()
    plt.plot(loss, color='b', label='training loss')
    plt.plot(v_loss, color='g', label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('Wykres funkcji straty')
    plt.legend()
    plt.show()
