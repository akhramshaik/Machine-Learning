# Machine Learning-> Classification: DecisionTreeClassifier

Lets look into parameters:
DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

max_depth: This defines how deep is our tree. The more we increase the depth the more overfitting happnes and the more complexity of the model increases. If we reduce the depth more also under fitting happens. This shuld be handeled by Hyper Parameter tuning to find best suite for this depth by GridSearchCv.


min_samples_split:This is nothng but the no of minimum samples required to continue your tree split. Each Tree will have Internal Nodes, Leaf Nodes. If you are at one of the Internal nodes and if you want to split your Nodes further deeper then this condidtion will get applied. If your min_samples_split=3 then your Internal Node should contain atleast >=3 samples. If you increase your min_samples_split value You would decrease the depth of your tree, This is because you would run out of samples to split and this would reduce overfitting.

min_samples_leaf: This is the minimum samples required to stop your tree growth further. min_samples_split is used to decide if ur tree can be broken further or not where as min_samples_leaf is to decide to stop the tree if the limit is reached.