from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping


class NeuralNetworkClassifier():

    def plot_history(self, history):
        import matplotlib.pyplot as plt
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(val_loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        print("=======================")
        print("val_loss ",val_loss_values)
        print("loss ", loss_values)

    def __init__(self, dim_features, nb_attributes=85, batch_size=200, nb_epoch=150):
        inputs = Input(shape=(dim_features,))
        x = Dense(700, activation='relu')(inputs)

        predictions = []
        for p in range(nb_attributes):
            predictions.append(Dense(1, activation='sigmoid')(x))

        self.model = Model(inputs, predictions)

        self.model.compile(optimizer=optimizers.Adam(lr=0.001), loss=['binary_crossentropy'] * nb_attributes)
        self.earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='auto')
        self.callbacks_list = [self.earlystop]
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
        h = self.model.fit(X, list(y.T), self.batch_size, self.nb_epoch, validation_split=0.2)
        self.plot_history(h)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return 0
