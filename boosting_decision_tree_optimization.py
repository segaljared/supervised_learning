import sklearn.tree
import sklearn.ensemble
import complexity_analysis
import numpy

def optimize_boosting_decision_tree(training_features, training_classes, problem_name, depth_range, learning_rate_range):
    def create_boosting_decision_tree(depth):
        return sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=depth))
    max_depth = complexity_analysis.run_complexity_analysis('Boosting Decision Tree Max Depth for %s' % problem_name, 
                                                            'Max Depth', 
                                                            'Accuracy', 
                                                            create_boosting_decision_tree, 
                                                            depth_range, 
                                                            training_features, 
                                                            training_classes, 
                                                            complexity_analysis.general_accuracy,
                                                            folds=5)

    def create_boosting_decision_tree_learning_rate(learning_rate):
        return sklearn.ensemble.AdaBoostClassifier(learning_rate=learning_rate, base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=max_depth))
    learning_rate = complexity_analysis.run_complexity_analysis('Boosting Decision Tree Learning Rate for %s' % problem_name, 
                                                            'Learning Rate', 
                                                            'Accuracy', 
                                                            create_boosting_decision_tree_learning_rate, 
                                                            learning_rate_range, 
                                                            training_features, 
                                                            training_classes, 
                                                            complexity_analysis.general_accuracy,
                                                            folds=5)
    estimator = sklearn.ensemble.AdaBoostClassifier(learning_rate=learning_rate, base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=max_depth))
    complexity_analysis.run_learning_curve_analysis(problem_name + ' Boosting Decision Tree', estimator, training_features, training_classes)
    return estimator