from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class Model:

    def __init__(self, tokenizer_len, X_train, X_test, y_train, y_test):
        self.tokenizer_len = tokenizer_len
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def run_model(self):
        self.model = self.create_model()
        self.fit_model()
        self.evaluate_model()

    def create_model(self):
        model = Sequential()

        model.add(Embedding(self.tokenizer_len, 64))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.summary()
        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def fit_model(self):
        # Fit the model
        history = self.model.fit(self.X_train, self.y_train,
                            batch_size=50, epochs=10)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_test = np.argmax(self.y_test, axis=1)

        predictions = []
        for i in range(len(y_pred)):
            if y_pred[i][0] > y_pred[i][1]:
                predictions.append(0)
            elif y_pred[i][0] <= y_pred[i][1]:
                predictions.append(1)

        predictions = np.array(predictions)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print("---------------")
        print("Precision score = ", round(precision, 3))
        print("Recall score = ", round(recall, 3))
        print("Accuracy score = ", round(accuracy, 3))
        print("F1 score = ", round(f1, 3))

        confusionMatrix = confusion_matrix(y_test, predictions)
        print("---------------")
        print("True negative : ", confusionMatrix[0][0])
        print("False negative : ", confusionMatrix[0][1])
        print("True positive : ", confusionMatrix[1][1])
        print("False positive : ", confusionMatrix[1][0])

        disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
        disp.plot()
        plt.show()