import numpy as np
from utils import bzUnpickle, get_class_attributes, create_data
from NeuralNetworkClassifier import NeuralNetworkClassifier
from NeuralNetworkRegressor import NeuralNetworkRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy

def DirectAttributePrediction(predicate_type='binary'):
    # Get features index to recover samples
    train_index = bzUnpickle('./CreatedData/train_features_index.txt')
    test_index = bzUnpickle('./CreatedData/test_features_index.txt')
    val_index=bzUnpickle('./CreatedData/validation_features_index.txt')
    # Get classes-attributes relationship
    train_attributes = get_class_attributes('./Classes/', name='train', predicate_type=predicate_type)
    test_attributes = get_class_attributes('./Classes/', name='test', predicate_type=predicate_type)
    N_ATTRIBUTES = train_attributes.shape[1]

    # Create training Dataset
    print('Creating training dataset...')
    X_train, y_train = create_data('./CreatedData/train_featuresVGG19.pic.bz2', train_index, train_attributes)



    print('Creating seen test dataset...')
    X_test_seen, y_test_seen = create_data('./CreatedData/validation_featuresVGG19.pic.bz2', val_index, train_attributes)
    y_pred_ = np.zeros(y_test_seen.shape)
    y_proba_ = np.copy(y_pred_)

    print('X_train to dense...')
    X_train = X_train.toarray()

    print('X_test_seen to dense...')
    X_test_seen = X_test_seen.toarray()

    print('Creating test dataset...')
    X_test, y_test = create_data('./CreatedData/test_featuresVGG19.pic.bz2', test_index, test_attributes)
    y_pred = np.zeros(y_test.shape)
    y_proba = np.copy(y_pred)

    print('X_test to dense...')
    X_test = X_test.toarray()


    if predicate_type != 'binary':
        clf = NeuralNetworkRegressor(dim_features=X_train.shape[1], nb_attributes=N_ATTRIBUTES)
    else:
        clf = NeuralNetworkClassifier(dim_features=X_train.shape[1], nb_attributes=N_ATTRIBUTES)

    print('Fitting Neural Network...')
    # fix random seed for reproducibility
    # seed = 7
    # numpy.random.seed(seed)
    # X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=1, random_state=seed)
    his=clf.fit(X_train, y_train)

    print('Predicting attributes...')
    y_pred = np.array(clf.predict(X_test))
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1])).T
    y_proba = y_pred

    y_pred_ = np.array(clf.predict(X_test_seen))
    y_pred_ = y_pred_.reshape((y_pred_.shape[0], y_pred_.shape[1])).T
    y_proba_ = y_pred_

    print('Saving files...')
    np.savetxt('./DAP_' + predicate_type + '/prediction_NN', y_pred)
    np.savetxt('./DAP_' + predicate_type + '/xprediction_NN', y_pred_)
    if predicate_type == 'binary':
        np.savetxt('./DAP_' + predicate_type + '/probabilities_NN', y_proba)
        np.savetxt('./DAP_' + predicate_type + '/xprobabilities_NN', y_proba_)


