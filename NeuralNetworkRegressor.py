from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras import optimizers
from keras import regularizers
class NeuralNetworkRegressor():                                                  
    # plot history
    def plot_history(self,history):
        import matplotlib.pyplot as plt
        history_dict = history.history
        loss_values = history_dict['loss']
        acc_values = history_dict['acc']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(acc_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.clf()
        val_acc_values = history_dict['val_acc']
        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        

    def __init__(self, dim_features, nb_attributes=85, batch_size=10, nb_epoch=10):
        inputs = Input(shape=(dim_features,))
        x = Dense(512, activation='tanh',
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(inputs)
        x = Dropout(0.3)(x)
        predictions = []
        for p in range(nb_attributes):
            predictions.append(Dense(1, activation='linear',
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x))
        self.model = Model(inputs, predictions)
        self.model.compile(optimizer="adam", loss=['mae'] * nb_attributes,metrics=['accuracy'])
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
       return self.model.fit(X, list(y.T), self.batch_size, self.nb_epoch,validation_split=0.2)


    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return 0
