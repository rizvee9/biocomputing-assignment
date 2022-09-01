import numpy as np

# It's a neural network with one hidden layer that uses the sigmoid activation function
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn() for i in range(6)])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        """
        The sigmoid function takes in a number and returns a number between 0 and 1

        :param x: the input data
        :return: The sigmoid function is being returned.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        """
        The derivative of the sigmoid function is the sigmoid function times one minus the sigmoid
        function

        :param x: The input to the sigmoid function
        :return: The derivative of the sigmoid function.
        """
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        """
        We take the dot product of the input vector and the weights vector, add the bias, and then pass
        the result through the sigmoid function

        :param input_vector: The input vector that we want to predict
        :return: The prediction of the input vector.
        """
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        """
        We compute the gradient of the error with respect to the bias and weights

        :param input_vector: The input vector to the neural network
        :param target: The target value we're trying to predict
        :return: The gradient of the error with respect to the bias and the gradient of the error with
        respect to the weights.
        """
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derrorpred = 2 * (prediction - target)
        dpredlay1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derrorpred * dpredlay1 * dlayer1_dbias
        )
        derror_dweights = (
            derrorpred * dpredlay1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        """
        We update the weights and biases of the layer by subtracting the product of the learning rate
        and the derivative of the error with respect to the weights and biases

        :param derror_dbias: The derivative of the error with respect to the bias
        :param derror_dweights: The derivative of the error with respect to the weights
        """
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        """
        We loop through all the instances in the training data, and for each instance, we compute the
        gradients and update the weights

        :param input_vectors: The input vectors for the training data
        :param targets: The target values for each of the data instances
        :param iterations: The number of times we want to train the model
        :return: The cumulative error for each iteration.
        """
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
