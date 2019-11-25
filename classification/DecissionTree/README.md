# Machine-Learning
This Repo is about Machine Learning Tuts.

Lets look into parameters:
DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

<b>max_depth:</b> This defines how deep is our tree. The more we increase the depth the more overfitting happnes and the more complexity of the model increases. If we reduce the depth more also under fitting happens. This shuld be handeled by Hyper Parameter tuning to find best suite for this depth by GridSearchCv.
