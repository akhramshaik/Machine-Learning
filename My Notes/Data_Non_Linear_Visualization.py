import sys
sys.path.append('C:/Users/akhram/Desktop/AIML/Machine Learning/Utils')

import classification_utils as cutils
from sklearn import model_selection

X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)