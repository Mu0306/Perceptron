import matplotlib.pyplot as plt
import numpy as np

class Myperceptron:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_rate = 1

    def fit(self,X_train,y_train):
        self.w = np.zeros(X_train.shape[1])
        i = 0
        while i < X_train.shape[0]:
            X = X_train[i]
            y = y_train[i]

            if y*(np.dot(self.w,X)+self.b) <= 0:
                self.w = self.w+self.l_rate * np.dot(y,X)
                self.b = self.b+self.l_rate * y
                i = 0
            else:
                i = i+1
def draw(X,w,b):
    X_new = np.array([[0],[6]])
    y_new = -b-(w[0]*X_new/w[1])
    plt.plot(X[:2,0],X[:2,1],"g*",label="A")
    plt.plot(X[2:,0],X[2:,1],"r*",label="B")
    plt.plot(X_new,y_new,"b-")
    plt.axis([0,6,0,6])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def main():
    X_train = np.array([[3,3],[4,3],[1,1]])
    y_train = np.array([1,1,-1])
    perceptron = Myperceptron()
    perceptron.fit(X_train,y_train)
    print(perceptron.w)
    print(perceptron.b)
    draw(X_train,perceptron.w,perceptron.b)

if __name__ == "__main__":
    main()