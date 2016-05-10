import matplotlib.pyplot as plt
from sklearn import datasets, svm, linear_model, metrics, cross_validation, grid_search

# Digits dataset
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(digits.images, digits.target, test_size=0.7)


# Flatten all the training set images convert each row's 8*8
# 'inner' matrix into a 1d vector of 64 elements, so that the
# whole training set becomes a matrix of the form
# (number of samples (5000*0.3), number of features (64)):
n_train_samples = len(X_train)
X_train_flat = X_train.reshape((n_train_samples, -1))
print('Dim(X_train_flat): {}'.format(X_train_flat[:n_train_samples].shape))


# Define class
# ifiers to compare
# classifier = grid_search.GridSearchCV(cv=None,
#        estimator=linear_model.LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
#           penalty='l2', tol=0.0001),
#        param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train_flat[:n_train_samples], y_train.ravel())


# Predict on the test set. Flatten test set images
# the same way we did with the training images
n_test_samples = len(X_test)
X_test_flat = X_test.reshape((n_test_samples, -1))
print('Dim(X_test_flat): {}\n'.format(X_test_flat.shape))
predicted = classifier.predict(X_test_flat)


# Print reports
score = classifier.score(X_test_flat, y_test)
print('Classification score: {:.2%}\n'.format(score))
print('Classification report for classifier %s:\n%s\n'
      % (classifier, metrics.classification_report(y_test, predicted)))
print('Confusion matrix:\n%s' % metrics.confusion_matrix(y_test, predicted))

# Plot results
images_and_predictions = list(zip(X_test, predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:10]):
    plt.subplot(1, 10, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title('Predict: %i' % prediction, fontsize=8)

plt.show()
