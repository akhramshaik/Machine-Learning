import sys
sys.path.append('C:/Users/akhram/Desktop/AIML/Machine Learning/Utils')

import classification_utils as cutils
from sklearn import model_selection

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=500, n_features=2, n_classes=2, weights=[0.4, 0.6], class_sep=1)
cutils.plot_data_2d_classification(X, y)


X, y = cutils.generate_linear_synthetic_data_classification(n_samples=500, n_features=3, n_classes=3, weights=[0.4, 0.3,0.3], class_sep=1)
cutils.plot_data_3d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

