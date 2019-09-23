import sklearn.neighbors
import complexity_analysis


def optimize_nearest_neighbors(training_features, training_classes, problem_name, k_range, p_range):
    def create_nearest_neighbor_k(k):
        return sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
    k = complexity_analysis.run_complexity_analysis('K Nearest Neighbors for %s' % problem_name,
                                                    '# of Neighbors', 
                                                    'Accuracy', 
                                                    create_nearest_neighbor_k, 
                                                    k_range, 
                                                    training_features, 
                                                    training_classes, 
                                                    complexity_analysis.general_accuracy,
                                                    folds=5)

    def create_nearest_neighbors_p(p):
        return sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
    p = complexity_analysis.run_complexity_analysis('Nearest Neighbors P for %s' % problem_name,
                                                    'P in Distance', 
                                                    'Accuracy', 
                                                    create_nearest_neighbors_p, 
                                                    p_range, 
                                                    training_features, 
                                                    training_classes, 
                                                    complexity_analysis.general_accuracy,
                                                    folds=5)

    estimator = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
    complexity_analysis.run_learning_curve_analysis(problem_name + ' Nearest Neighbors', estimator, training_features, training_classes)
    return estimator
