from sklearn.impute import SimpleImputer
import sklearn.model_selection as sklearn_m
import numpy
import math
import csv
import decision_tree_optimization
import boosting_decision_tree_optimization
import nearest_neighbors_optimization
import neural_net_optimization
import svm_opmtimization
import complexity_analysis
import time
import matplotlib.pyplot as plot


def main():
    default_credit()
    census()

def default_credit():
    features, classes = load_default_data()
    test_size = int(len(features) - 5000)
    training_features, test_features, training_classes, test_classes = sklearn_m.train_test_split(features, classes, test_size=test_size, train_size=(len(features) - test_size), random_state=50207)
    
    decision_tree = decision_tree_optimization.optimize_decision_tree(training_features, training_classes, 'CC Default', range(1, 25), range(1, len(training_features[0])))
    knn = nearest_neighbors_optimization.optimize_nearest_neighbors(training_features, training_classes, 'CC Default', range(1, 20), range(1, 5))
    boosting_trees = boosting_decision_tree_optimization.optimize_boosting_decision_tree(training_features, training_classes, 'CC Default', range(1, 10), numpy.linspace(0.5, 1.5, 7))
    neural_net = neural_net_optimization.optimize_neural_network(training_features, training_classes, 'CC Default', range(5, 55, 10))
    svm = svm_opmtimization.optimize_svm(training_features, training_classes, 'CC Default', [0.001, 0.01, 0.1, 1.0], [0.001, 0.01, 0.1, 1.0, 10.0], range(1, 6))

    train_and_predict_and_plot('CC Default',
                                [decision_tree, knn, boosting_trees, neural_net, svm],
                                ['DT', 'kNN', 'BDT', 'NN', 'SVM'],
                                training_features,
                                training_classes,
                                test_features,
                                test_classes)

def census():
    features, classes = load_census_data()
    test_size = int(len(features) - 5000)
    training_features, test_features, training_classes, test_classes = sklearn_m.train_test_split(features, classes, test_size=test_size, train_size=(len(features) - test_size), random_state=50207)

    training_features, training_classes = synced_shuffle(training_features, training_classes, 3643354)

    decision_tree = decision_tree_optimization.optimize_decision_tree(training_features, training_classes, 'Census Income', range(1, 25), range(1, len(features[0])))#numpy.linspace(0.05, 0.85, 9))
    knn = nearest_neighbors_optimization.optimize_nearest_neighbors(training_features, training_classes, 'Census Income', range(1, 20), range(1, 6))
    boosting_trees = boosting_decision_tree_optimization.optimize_boosting_decision_tree(training_features, training_classes, 'Census Income', range(1, 11), numpy.linspace(0.5, 1.5, 7))
    neural_net = neural_net_optimization.optimize_neural_network(training_features, training_classes, 'Census Income', range(5, 55, 10))
    svm = svm_opmtimization.optimize_svm(training_features, training_classes, 'Census Income', [0.001, 0.01, 0.1, 1.0], [0.001, 0.01, 0.1, 1.0, 10.0], range(1, 6))
    
    train_and_predict_and_plot('Census Income',
                                [decision_tree, knn, boosting_trees, neural_net, svm],
                                ['DT', 'kNN', 'BDT', 'NN', 'SVM'],
                                training_features,
                                training_classes,
                                test_features,
                                test_classes)

def synced_shuffle(features, classes, random_seed):
    numpy.random.seed(random_seed)
    indices = numpy.arange(1, len(features))
    numpy.random.shuffle(indices)
    return features[indices], classes[indices]

def load_census_data():
    features = []
    classes = []
    value_maps = [None,
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
            None,
            ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
            None,
            ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
            ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
            ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
            ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
            ["Female", "Male"],
            None,
            None,
            None,
            ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]]
    with open('adult.data', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 0:
                continue
            feature_row = []
            for col, value in enumerate(row[:(len(row) - 1)]):
                value_map = value_maps[col]
                if value.strip() == '?':
                    feature_row.append(numpy.nan)
                elif value_map is not None:
                    feature_row.append(float(value_map.index(value.strip())))
                else:
                    feature_row.append(float(value))
            features.append(feature_row)
            if row[len(row) - 1].strip() == '<=50K':
                classes.append(0.0)
            else:
                classes.append(1.0)
    imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    imputer.fit(features)
    features = imputer.transform(features)
    return numpy.array(features), numpy.array(classes)

def load_default_data():
    features = []
    classes = []
    with open('default_credit_card.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for csv_row in reader:
            class_index = len(csv_row) - 1
            row = csv_row[0:class_index]
            features.append(row)
            classes.append(csv_row[class_index])
    return numpy.array(features).astype(float), numpy.array(classes).astype(float)

def train_and_predict(estimator, training_features, training_classes, test_features, test_classes):
    start = time.time()
    estimator.fit(training_features, training_classes)
    training_time = time.time() - start
    start = time.time()
    accuracy = complexity_analysis.general_accuracy(estimator, test_features, test_classes)
    predict_time = time.time() - start
    return accuracy, training_time, predict_time

def train_and_predict_and_plot(name, estimators, estimator_names, training_features, training_classes, test_features, test_classes):
    accuracy_all = []
    training_time_all = []
    predict_time_all = []
    for estimator in estimators:
        accuracy, training_time, predict_time = train_and_predict(estimator, training_features, training_classes    , test_features, test_classes)
        accuracy_all.append(accuracy)
        training_time_all.append(training_time)
        predict_time_all.append(predict_time)
    
    plot.clf()
    plot.xlabel('Algorithm Type')
    plot.ylabel('Test Accuracy')
    plot.scatter(estimator_names, accuracy_all, label='accuracy', c='darkorange')
    plot.savefig('%s_test_accuracy' % name.replace(' ', '_'))
    
    plot.clf()
    plot.xlabel('Algorithm Type')
    plot.ylabel('Time (s)')
    plot.scatter(estimator_names, training_time_all, label='training time', c='darkorange')
    plot.scatter(estimator_names, predict_time_all, label='predict time', c='navy', marker='x')
    plot.legend(loc='best')
    plot.pause(0.001)
    plot.savefig('%s_times' % name.replace(' ', '_'))

if __name__ == "__main__":
    main()