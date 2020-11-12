import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class averagedPerceptron: 
    def __init__(self, iter):
        self.iter =iter

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        w = np.zeros(x_train.shape[1])
        b = 0
        u = np.zeros(x_train.shape[1])
        B = 0
        c = 1
        for _ in range(self.iter):
            for x, y in zip(x_train, y_train):
                if y * (np.dot(w, x) + b) <= 0:
                    w += y * x
                    b += y
                    u += y * c * x
                    B += y * c
                c += 1
        w = np.array([w - ((1/c) * u)])
        b = np.array([b - ((1/c) * B)])
        model = np.append(b,w)
        return model

    def predict(self, newdata, model):
        newdata2 = np.hstack((np.ones((len(newdata),1)), newdata))
        y_pred = np.sign(np.dot(newdata2, model))
        return y_pred

    def score(self, y_pred, y_test):
        return np.sum(np.equal(y_pred, y_test))/len(y_pred)

data = load_files('review_polarity/txt_sentoken/')
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(data.data)

x_train, x_test, y_train, y_test = train_test_split(vector.toarray(), data.target, test_size=0.2)

y_test = np.where(y_test==0, -1, y_test)
y_train = np.where(y_train==0, -1, y_train)

model1 = averagedPerceptron(10)
trainedModel = model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test,trainedModel)
print('Accuracy average perceptron 10 iteration: %.5f' % model1.score(y_pred1,y_test))

model2 = averagedPerceptron(100)
trainedModel = model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test,trainedModel)
print('Accuracy average perceptron 100 iteration: %.5f' % model1.score(y_pred2,y_test))

model3 = averagedPerceptron(1000)
trainedModel = model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test,trainedModel)
print('Accuracy average perceptron 1000 iteration: %.5f' % model1.score(y_pred3,y_test))