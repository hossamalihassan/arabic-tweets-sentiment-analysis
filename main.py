from dataHandler import DataHandler
from preprocessing import Preprocessing
from model import Model

dataHandler = DataHandler()
(train_data, test_data, data) = dataHandler.get_data()

preprocessing = Preprocessing(train_data, test_data, data)
(X_train, X_test, y_train, y_test) = preprocessing.get_x_y()

model = Model(preprocessing.get_tokenizer_len(), X_train, X_test, y_train, y_test)
model.run_model()