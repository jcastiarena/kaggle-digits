import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm, metrics, cross_validation, preprocessing

import time
start = time.time()

def load_data(file, n_rows=None, skip_rows=None):
    data = pd.read_csv(file, header=0, index_col=None, low_memory=False, nrows=n_rows, skiprows=skip_rows)
    print('\t{}: {}'.format(file, data))
    print('\tDim({}): {}'.format(file, data.shape))
    is_train = 'train' == file.split('.')[0]
    if is_train:
        X = data.ix[:, range(1, 785)].values
        y = data.ix[:, 0].values
        print('\tX_train: \n{}'.format(X))
        print('\tDim(X_train): {}'.format(X.shape))
        print('\ty_train: {}'.format(y))
        print('\tDim(y_train): {}'.format(y.shape))
    else:
        X = data.ix[:, range(0, 784)].values
        y = np.zeros(len(data), dtype=np.int)
        print('\tX_test: \n{}'.format(X))
        print('\tDim(X_test): {}'.format(X.shape))
        print('\ty_test: {}'.format(y))
        print('\tDim(y_test): {}'.format(y.shape))

    return data, X, y


# Load train data
print('Loading data...')
# digits_train, X_train, y_train = load_data('train.csv', n_rows=4200, skip_rows=10)
# digits_test, X_test, y_test = load_data('test.csv', n_rows=2800, skip_rows=10)
digits_train, X_train, y_train = load_data('train.csv')
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train, y_train, test_size=0.3)


n_train_samples = len(X_train)


# Define classifiers to compare
# classifier = grid_search.GridSearchCV(cv=None,
#        estimator=linear_model.LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
#           penalty='l2', tol=0.0001),
#        param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})


# Scale training data
X_train_scaled = preprocessing.scale(X_train)
print('\tX_train_scaled: {}'.format(X_train_scaled))
print('\tDim(X_train_scaled): {}'.format(X_train_scaled.shape))


# Train
classifier = svm.SVC(gamma=0.001)
print('Training {}...'.format(classifier))
classifier.fit(X_train_scaled, y_train)



# Predict on the test set
print('Predicting...')
digits_test, X_test, y_test = load_data('test.csv')


# Scale test data
X_test_scaled = preprocessing.scale(X_test)
print('\tX_test_scaled: {}'.format(X_test_scaled))
print('\tDim(X_test_scaled): {}'.format(X_test_scaled.shape))

n_test_samples = len(X_test)

# Predict
predicted = classifier.predict(X_test_scaled)
print('\tPredicted: {}'.format(predicted))
print('\tDim(predicted): {}'.format(predicted.shape))

# Print reports (calculate score on the second half of the training data)
# print('Printing reports...')
# score = classifier.score(X_train_flat[n_train_samples/2:], y_train.ravel()[n_labels_samples/2:])
# print('Classification score: {:.2%}\n'.format(score))
# print('Classification report for classifier %s:\n%s\n'
#       % (classifier, metrics.classification_report(y_test, predicted)))
# print('Confusion matrix:\n%s' % metrics.confusion_matrix(y_test, predicted))



# Plot results
print('Plotting info...')

X_test_reshaped = X_test.reshape(n_test_samples, 28, 28)
print('\tDim(X_test): {}'.format(X_test.shape))
print('\tDim(X_test_reshaped): {}'.format(X_test_reshaped.shape))
print('\tX_test_reshaped: {}'.format(X_test_reshaped))

images_and_predictions = list(zip(X_test_reshaped, predicted))
print('\tlen(images_and_predictions): {}'.format(len(images_and_predictions)))
print('\tlen(predicted): {}'.format(len(predicted)))
for index, (image, prediction) in enumerate(images_and_predictions[:10]):
    # print('Image: {}'.format(image))
    # print('Prediction: {}'.format(prediction))

    plt.subplot(1, 10, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title('Predict: %i' % prediction, fontsize=8)


# Create a new dataframe for Kaggle
print('Generating output file...')
submission = pd.DataFrame({
        "ImageId": np.arange(1, len(images_and_predictions) + 1),
        "Label": predicted
    })

submission.to_csv("kaggle-digits.csv", index=False)

print('Script took {} secs to run'.format((time.time() - start) ))


plt.show()


