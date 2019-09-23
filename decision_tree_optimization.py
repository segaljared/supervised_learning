import sklearn.tree
import complexity_analysis
import numpy

def optimize_decision_tree(training_features, training_classes, problem_name, depth_range, number_features_range, scorer=complexity_analysis.general_accuracy):
    def create_decision_tree(depth):
        return sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    max_depth = complexity_analysis.run_complexity_analysis('Decision Tree Max Depth for %s' % problem_name, 
                                                            'Max Depth', 
                                                            'Accuracy', 
                                                            create_decision_tree, 
                                                            depth_range, 
                                                            training_features, 
                                                            training_classes, 
                                                            scorer,
                                                            folds=5)

    def create_decision_tree_max_features(number_features):
        return sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, max_features=number_features)
    max_features = complexity_analysis.run_complexity_analysis('Decision Tree Max Features for %s' % problem_name, 
                                                            'Max Features to Examine', 
                                                            'Accuracy', 
                                                            create_decision_tree_max_features, 
                                                            number_features_range, 
                                                            training_features, 
                                                            training_classes, 
                                                            scorer,
                                                            folds=5)
    estimator = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    complexity_analysis.run_learning_curve_analysis(problem_name + ' Decision Tree', estimator, training_features, training_classes)
    return estimator