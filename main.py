import numpy as np


class HandWritterDigitRecognition:
    def __init__(self):
        self.train_data_file = "sample_data/mnist_train.csv"
        self.test_data_file = "sample_data/mnist_test.csv"

    def get_data_from_train_data_file(self):
        data = np.genfromtxt(
            fname=self.train_data_file, delimiter=",", dtype=int
        )
        print("Shape of data in train data file is {}".format(data.shape))
        mnist_labels = data[:, 0]
        mnist_variables = data[:, 1:]
        return mnist_variables, mnist_labels

    @staticmethod
    def shuffle(variables, labels):
        data_count = variables.shape[0]
        np.random.seed(3)
        permute_indices = np.random.permutation(data_count)
        shuffled_variables = variables[permute_indices]
        shuffled_labels = labels[permute_indices]
        return shuffled_variables, shuffled_labels

    def get_best_k_n_values_using_validation_set(
            self, variables, labels, validation_split_percent
    ):
        import math
        shuffled_variables, shuffled_labels = \
            self.shuffle(variables=variables, labels=labels)
        train_data_count = math.floor(
            (float(100 - validation_split_percent) / 100) * variables.shape[0]
        )
        train_inputs = shuffled_variables[:train_data_count]
        train_outputs = shuffled_labels[:train_data_count]
        validation_inputs = shuffled_variables[train_data_count:]
        validation_outputs = shuffled_labels[train_data_count:]

    def distance_weighted_knn(self, train_inputs, train_outputs, test_inputs, n, k):
        """
        Predict the label using distance weighted KNN

        :param train_inputs: a 2D numpy array of floats where each row represents a training input instance
        :param train_outputs: a 2D numpy array that represents the labels corresponds to train_inputs
        :param test_inputs: a 2D numpy array of floats which represent training instances
        :param n: n is for compute LN Norm distance
        :param k: k is the number of closest neighbours to consider
        :return:
        """
        unique_class_labels = np.unique(train_outputs)
        weights = np.zeros(shape=(train_inputs.shape[0], unique_class_labels.shape[0]))
        for test_idx, test_input in enumerate(test_inputs):
            k_distance_indices, k_distances = self.k_nearest_neightbours(
                train_inputs=train_inputs, test_input=test_input, n=n, k=k
            )
            predicted_labels = train_outputs[k_distance_indices]
            for label_idx, label in enumerate(unique_class_labels):
                label_weight = np.sum(np.where(predicted_labels == label, 1/k_distances, 0.0))
                weights[test_idx][label_idx] = label_weight

        highest_label_indices = np.argmax(weights, axis=1)
        return unique_class_labels[highest_label_indices]

    def k_nearest_neightbours(self, train_inputs, test_input, n, k):
        """
        Get K nearest neighbours of the test inputs

        :param train_inputs: a 2D numpy array of floats where each row represents a training input instance
        :param test_input: a 1D numpy array of floats which represent training instance
        :param n: n is for compute LN Norm distance
        :param k: k is the number of closest neighbours to consider
        :return: returns indices of K-nearest neighbours and their distances
        """
        distances = self.ln_norm_distances(
            train_inputs=train_inputs, test_input=test_input, n=n
        )
        indices = np.argsort(distances)
        kth_dist_repeat_count = 0
        if train_inputs.shape[0] > k:
            kth_nearesh_neighbour_index = indices[k - 1]  # last most neighbour
            kth_neighbour_distance = distances[kth_nearesh_neighbour_index]
            indices_except_top_k = indices[k:]
            # distance tie
            distance_of_points_except_top_k = distances[indices_except_top_k]
            kth_dist_repeat_count = np.count_nonzero(distance_of_points_except_top_k == kth_neighbour_distance)
        indices_of_k_neighbours = indices[:(k+kth_dist_repeat_count)]
        distance_k = distances[indices_of_k_neighbours]
        return indices_of_k_neighbours, distance_k

    @staticmethod
    def ln_norm_distances(train_inputs, test_input, n):
        """
        LN Norm Distances which computes distances between
        a testing instance and other training instance

        :param train_inputs: a 2D numpy array of floats where each row represents a training input instance
        :param test_input: a 1D numpy array of floats which represent training instance
        :param n: n is for compute LN Norm distance
        :return: a 1D of array floats i.e distance between testing instance and trainging instances
        """
        abs_diff = np.abs(train_inputs - test_input)
        summation = np.sum(np.power(abs_diff, n), axis=1)
        return np.power(summation, 1 / n)
