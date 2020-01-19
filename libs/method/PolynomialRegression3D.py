import sys
import math
import logging
import tensorflow as tf
import numpy as np
from libs.model.PolynomialParam import PolynomialParam

# model parameters
polynomial_degree = 4  # order of polynomial


class PolynomialRegression3D:
    def __init__(self, order):
        self.poly_degree = order
        self.lambda_values = [1E-5, 1E-4, 1E-3]
        self.random_init_num = 25
        self.best_order = 0

    def get_number_of_params(self, order):
        count = 0
        for i in range(order):
            for j in range(order - i):
                for k in range(order - i - j):
                    count += 1
        return count

    def train(self, a, b, c, y):
        print('training...')

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        y = np.array(y)

        combined = np.stack((a, b, c, y), axis=-1)

        best_lambda = 1E-3
        best_params = None
        min_error = sys.maxsize

        for j in range(len(self.lambda_values)):
            params, error = self.cross_validation(
                combined, self.lambda_values[j], polynomial_degree, 1E1, 1E10)

            if error < min_error:
                best_params = params
                min_error = error
                best_lambda = self.lambda_values[j]

        print('Best Lambda: ' + str(best_lambda))

        params, error = self.polynomial_regression(
            a, b, c, y, best_lambda, polynomial_degree, 1E1, 1E-6)

        if error < min_error:
            best_params = params

        return best_params, error

    def cross_validation(self, combined, lambda_value, order, learning_rate, escape_rate):
        np.random.shuffle(combined)
        length = combined.shape[0]
        training_count = int(math.floor(length * 0.7))
        splited = np.split(combined, [training_count, length])

        # separate training data to 7, 3 for cross validation
        a_train, b_train, c_train, y_train = splited[0].T
        a_validate, b_validate, c_validate, y_validate = splited[1].T

        params, _ = self.polynomial_regression(
            a_train, b_train, c_train, y_train, lambda_value, order, learning_rate, escape_rate)
        error = self.calculate_total_error(
            a_validate, b_validate, c_validate, y_validate, params, order)
        error = error / a_validate.shape[0]
        return params, error

    def polynomial_regression(self, input_a, input_b, input_c, y, lambda_value, order, learning_rate, escape_rate):

        number_of_param = self.get_number_of_params(order)

        y_input = np.array(y)
        y_input = y_input.reshape((y_input.size, 1))

        # random initialization
        W = tf.Variable(tf.random_uniform(
            [number_of_param, 1]), name='weight')
        b = tf.Variable(tf.random_uniform([1]), name='bias')  # bias

        X = tf.placeholder(tf.float32, shape=[None, number_of_param])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        # hypothesis
        x_modified, x_scalar, x_power = self.get_x_modified_and_scalar(
            input_a, input_b, input_c, y_input.size, order)
        Y_pred = tf.add(tf.matmul(X, W), b)

        # cost function
        cost_function = tf.reduce_mean(
            tf.square(Y_pred - Y)) + tf.multiply(lambda_value, tf.global_norm([W]))

        # training algorithm
        optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(cost_function)

        best_params = None
        min_error = sys.maxsize

        for j in range(self.random_init_num):

            # initializing the variables
            init = tf.global_variables_initializer()

            # starting the session session
            sess = tf.Session()
            sess.run(init)

            epoch = 60000
            prev_training_cost = 0.0
            for step in range(epoch):
                _, training_cost = sess.run([optimizer, cost_function], feed_dict={
                    X: x_modified, Y: y_input})

                if np.abs(prev_training_cost - training_cost) < escape_rate:
                    break
                prev_training_cost = training_cost

            if min_error > prev_training_cost:
                weight = sess.run(W).ravel().tolist()
                bias = sess.run(b)[0].item()
                scaled_weight = [x / y for x, y in zip(weight, x_scalar)]
                best_params = PolynomialParam(scaled_weight, bias)
                min_error = prev_training_cost

        return best_params, min_error

    def predict(self, a, b, c, params, order=polynomial_degree):
        result = 0.0
        count = 0
        for i in range(order):
            for j in range(order - i):
                for k in range(order - i - j):
                    result += np.power(a, i) * \
                        np.power(b, j) * np.power(c, k) * params.W[count]
                    count += 1
        result += params.b
        return result

    def get_x_modified_and_scalar(self, a, b, c, size, order):

        number_of_param = self.get_number_of_params(order)

        # preparing the data
        x_modified = np.zeros([size, number_of_param])
        x_scalar = []
        x_power = np.zeros([number_of_param])
        count = 0

        # hypothsis
        for i in range(order):
            for j in range(order - i):
                for k in range(order - i - j):
                    x_modified[:, count] = np.power(
                        a, i) * np.power(b, j) * np.power(c, k)
                    max_x = np.max(x_modified[:, count])
                    x_modified[:, count] = x_modified[:, count] / max_x
                    x_scalar.append(max_x)
                    x_power[count] = i + j + k + 1
                    count += 1

        return x_modified, x_scalar, x_power

    def calculate_total_error(self, a, b, c, y, params, order):
        total_error = 0
        for i in range(a.size):
            error = self.predict(a[i], b[i], c[i], params, order) - y[i]
            total_error += error * error
        return total_error
