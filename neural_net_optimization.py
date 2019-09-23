import sklearn.neural_network
import complexity_analysis
import numpy
import matplotlib.pyplot as plot


def optimize_neural_network(training_features, training_classes, problem_name, hidden_layer_sizes):
    def create_neural_net_hidden_units(hidden_units):
        return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(hidden_units,), max_iter=10000)
    hidden_units = complexity_analysis.run_complexity_analysis('Neural Net Hidden Units for %s' % problem_name,
                                                            'Hidden Units',
                                                            'Accuracy',
                                                            create_neural_net_hidden_units,
                                                            hidden_layer_sizes,
                                                            training_features,
                                                            training_classes,
                                                            complexity_analysis.general_accuracy,
                                                            folds=5)

    hidden_units = 5
    learning_rate = 0.0001
    def create_neural_net():
        return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(hidden_units,), activation='logistic', max_iter=10000, solver='sgd', learning_rate='constant', learning_rate_init=0.0001, batch_size=10)
    
    weight_updates(create_neural_net, training_features, training_classes, max_iterations=50)
    plot.savefig('%s_nn_weight_updates' % problem_name.replace(' ', '_').lower())
    complexity_analysis.run_learning_curve_analysis(problem_name + ' NN', create_neural_net(), training_features, training_classes)
    return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(hidden_units,), learning_rate_init=learning_rate, max_iter=10000)

def weight_updates(create_neural_net, features, classes, max_iterations=10000, folds=5):
    fold_indices = []
    fold_size = int(len(classes) / folds)
    fold_start = 0
    for i in range(0, folds):
        fold_end = min(fold_start + fold_size, len(features) - 1)
        fold_indices.append(range(fold_start, fold_end))
        fold_start = fold_end + 1

    training_sets = []
    training_classes = []
    validation_sets = []
    validation_classes = []
    neural_nets = []
    for i in range(0, folds):
        training_indices = []
        validation_indices = fold_indices[i]
        for j in range(0, folds):
            if i != j:
                training_indices.extend(fold_indices[j])
        training_indices = numpy.array(training_indices).astype(int)
        training_sets.append(features[training_indices])
        training_classes.append(classes[training_indices])
        validation_sets.append(features[validation_indices])
        validation_classes.append(classes[validation_indices])
        neural_nets.append(create_neural_net())

    for i, nn in enumerate(neural_nets):
        nn.partial_fit(training_sets[i], training_classes[i], numpy.unique(classes))

    plot.ion()

    training_scores_means = []
    training_scores_stds = []
    validation_scores_means = []
    validation_scores_stds = []
    weight_updates = []

    for update in range(0, max_iterations):
        print(update)
        training_scores = []
        validation_scores = []
        neural_nets[0].partial_fit(training_sets[0], training_classes[0])
        for i, nn in enumerate(neural_nets):
            nn.partial_fit(training_sets[i], training_classes[i])
            if update % 10 == 0 and update > 1:
                training_scores.append(complexity_analysis.general_accuracy(nn, training_sets[i], training_classes[i]))
                validation_scores.append(complexity_analysis.general_accuracy(nn, validation_sets[i], validation_classes[i]))
        if len(training_scores) > 0:
            weight_updates.append(update)
            training_scores_means = numpy.append(training_scores_means, numpy.mean(training_scores))
            training_scores_stds = numpy.append(training_scores_stds, numpy.std(training_scores))
            validation_scores_means = numpy.append(validation_scores_means, numpy.mean(validation_scores))
            validation_scores_stds = numpy.append(validation_scores_stds, numpy.std(validation_scores))
            plot_neural_net_weight_updates_curve(weight_updates, training_scores_means, training_scores_stds,
                                                 validation_scores_means, validation_scores_stds)


def plot_neural_net_weight_updates_curve(weight_updates, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std):
    plot.clf()
    plot.title('Accuracy vs weight updates')
    plot.xlabel('Number of weight updates')
    plot.ylabel('Accuracy')
    plot.fill_between(weight_updates, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                      alpha=0.2, color="darkorange", lw=2)
    plot.plot(weight_updates, train_scores_mean, label="Training score", color="darkorange", lw=2)
    plot.fill_between(weight_updates, validation_scores_mean - validation_scores_std,
                      validation_scores_mean + validation_scores_std, alpha=0.2, color="navy", lw=2)
    plot.plot(weight_updates, validation_scores_mean, label="Validation score", color="navy", lw=2)
    plot.legend(loc="best")
    plot.grid()
    plot.pause(0.001)