import pickle
from sklearn.model_selection import train_test_split
import model_zero
import model_one
import model_two

classes = ["Mydło", "A", "Ą", "B", "C", "Ć", "D", "E", "Ę", "F", "G", "H", "I", "J", "K", "L",
           "Ł", "M", "N", "Ń", "O", "Ó", "P", "R", "S", "Ś", "T", "U", "W", "Y", "Z", "Ź", "Ż"]
height = 32
width = 32

with open("bin_dataset", 'rb') as ds:
    dataset = pickle.load(ds, encoding='bytes')
    X = dataset[0]
    Y = dataset[1]
    x_temp, x_test, y_temp, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    x_train, x_eval, y_train, y_eval = train_test_split(x_temp, y_temp, test_size=0.15, random_state=1)
    model_zero.run(x_train, y_train, x_test, y_test, x_eval, y_eval)
    model_one.run(x_train, y_train, x_test, y_test, x_eval, y_eval)
    model_two.run(x_train, y_train, x_test, y_test, x_eval, y_eval)


