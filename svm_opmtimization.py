import sklearn.svm
import complexity_analysis


def optimize_svm(training_features, training_classes, problem_name, gamma_range, C_range, degree_range):
    def create_svc_gamma(gamma):
        return sklearn.svm.SVC(gamma=gamma)
    gamma = complexity_analysis.run_complexity_analysis('SVM gamma for %s' % problem_name,
                                                        'Gamma',
                                                        'Accuracy',
                                                        create_svc_gamma,
                                                        gamma_range,
                                                        training_features,
                                                        training_classes,
                                                        complexity_analysis.general_accuracy,
                                                        folds=5,
                                                        x_log_space=True)
    
    def create_svc_C(c):
        return sklearn.svm.SVC(gamma=gamma, C=c)
    C = complexity_analysis.run_complexity_analysis('SVM C for %s' % problem_name,
                                                        'C',
                                                        'Accuracy',
                                                        create_svc_C,
                                                        C_range,
                                                        training_features,
                                                        training_classes,
                                                        complexity_analysis.general_accuracy,
                                                        folds=5,
                                                        x_log_space=True)

    def create_svc_kernel(kernel):
        return sklearn.svm.SVC(kernel=kernel)
    complexity_analysis.run_complexity_analysis('SVM C for %s' % problem_name,
                                                        'kernel',
                                                        'Accuracy',
                                                        create_svc_kernel,
                                                        ['linear', 'rbf'],
                                                        training_features,
                                                        training_classes,
                                                        complexity_analysis.general_accuracy,
                                                        folds=5)
    
    complexity_analysis.run_learning_curve_analysis(problem_name + ' SVM', sklearn.svm.SVC(), training_features, training_classes)
    return sklearn.svm.SVC()
