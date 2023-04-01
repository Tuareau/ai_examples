import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

def f1_measure(y_test, y_pred):
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]: # true prediction
            if y_pred[i] == 0: # positive
                tp += 1
            else:              # negative
                tn += 1          
        else:                            # false prediction
            if y_pred[i] == 1: # negative
                fn += 1
            else:              # positive
                fp += 1
    f1_m = tp/(tp + (fn + fp) / 2)
    print(f"TP (true positive) = {tp}, TN (true negative) = {tn}")
    print(f"FP (false positive) = {fp}, FN (false negative) = {fn}")
    print(f"Acurracy = (TP + TN)/(TP + TN + FP + FN) = {(tn + tp)/(tn + tp + fp + fn)}")
    print(f"Error rate = (FP + FN)/(TP + TN + FP + FN) = {(fn + fp)/(tn + tp + fp + fn)}")
    print(f"Presicion = TP/(TP + FP) = {tp/(tp + fp)}")
    print(f"Recall = TP/(TP + FN) = {tp/(tp + fn)}")
    print(f"F1 measure = TP/(TP + (FN + FP) / 2) = {f1_m}\n")

    return f1_m

# Определение класса нейронной сети
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        # Инициализация весов
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    # Функция активации
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        y_hat = self.sigmoid(self.z3)
        return y_hat

    # Обратное распространение ошибки
    def backward(self, X, y, y_hat):
        delta4 = (y_hat - y) * self.sigmoid(self.z3) * (1 - self.sigmoid(self.z3))
        dW3 = np.dot(self.a2.T, delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)

        delta3 = np.dot(delta4, self.W3.T) * self.sigmoid(self.z2) * (1 - self.sigmoid(self.z2))
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid(self.z1) * (1 - self.sigmoid(self.z1))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # Обновление весов
        self.W1 -= 0.1 * dW1
        self.b1 -= 0.1 * db1
        self.W2 -= 0.1 * dW2
        self.b2 -= 0.1 * db2
        self.W3 -= 0.1 * dW3
        self.b3 -= 0.1 * db3

    # Тренировка модели
    def train(self, X, y, epochs):
        for i in range(epochs):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat)

    # Предсказание результатов
    def predict(self, X):
        y_hat = self.forward(X)
        y_pred = np.round(y_hat)
        return y_pred
    

def generate_test_mesh():
    h = 0.005
    testX, testY = np.meshgrid(np.arange(0, 1.0, h), np.arange(0, 1.0, h))
    return [testX, testY]

# генерация обучающей выборки
X_train = np.random.rand(200, 2)
y_train = np.round(np.logical_xor(X_train[:,0] > 0.5, X_train[:,1] > 0.5)).reshape(-1,1)

# инициализация нейросети
model = NeuralNetwork(input_dim=2, hidden_dim1=8, hidden_dim2=8, output_dim=1)

model.train(X_train, y_train, epochs=2000)

# генерация тестовой выборки
X_test = np.random.rand(50, 2)
y_test = np.round(np.logical_xor(X_test[:,0] > 0.5, X_test[:,1] > 0.5)).reshape(-1,1)

# для вычисления гиперплоскостей
#X_test_mesh = generate_test_mesh()
#X_test_mesh_zip = zip(X_test_mesh[0].ravel(), X_test_mesh[1].ravel())
#X_test_dotted = np.array([np.array(a) for a in X_test_mesh_zip])
#y_test = np.round(np.logical_xor(X_test_dotted[:,0] > 0.5, X_test_dotted[:,1] > 0.5)).reshape(-1,1)

# предсказание на тестовой выборке
y_pred = model.predict(X_test)
#y_pred = model.predict(X_test_dotted)


for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        y_pred[i] = 2 # ошибка распознавания

plt.figure()

#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])

# построение гиперплоскостей
#plt.pcolormesh(X_test_mesh[0],
#               X_test_mesh[1],
#               y_pred.reshape(X_test_mesh[0].shape),
#               cmap=ListedColormap(['#FF00FF','#AAAAAA','#DC143C' ]))

# выводим данные обучающей выборки на график
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['#6A5ACD', '#000000',]))

f1_measure(y_test, y_pred)

# выводим данные тестовой выборки на график
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=ListedColormap(['#FF00FF', '#AAAAAA', '#DC143C']))

plt.title('Recognizing Result')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()