import numpy as np
import collections
import matplotlib.pyplot as plt
import sklearn.gaussian_process as GP

data_description = collections.namedtuple("data_description", ("Inputs",
                                                               "Targets"))


class GaussianProcess(object):
    """
    Generates a batch of 1D GP curve samples.
    Supports RBF and periodic kernels and can switch between them halfway through.
    """

    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_dim=1,
                 y_dim=1,
                 l_scale=0.4,
                 sigma_scale=1,
                 period=0,
                 single_kernel=True,
                 testing=False):

        """
        :param batch_size: Integer; num of functions sampled
        :param max_num_context: Integer; max num of observations to train with
        :param x_dim: Integer; dimension of input vector x
        :param y_dim: Integer; dimension of output vector y
        :param l_scale: Float; length scale of kernel
        :param sigma_scale: Float; scale of variance
        :param period: Float; period of periodic kernel
        :param single_kernel: Boolean; indicates whether the kernel function is the same across the whole x dim
        :param testing: Boolean; Indicates whether testing, if so then want linspace of target vectors
        """

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._l_scale = l_scale
        self._sigma_scale = sigma_scale
        self._period = period
        self._single_kernel = single_kernel
        self._testing = testing

    def generate_curves(self):
        """
        :return: named tuple containing regression dataset
        """
        if self._testing:
            total_points = 400
        else:
            total_points = 50

        x_data = np.zeros(shape=(self._batch_size, total_points, 1), dtype=np.float32)
        y_data = np.zeros(shape=(self._batch_size, total_points, 1), dtype=np.float32)

        # sample N ~ uniform(1, max_num_context)
        num_context_points = np.random.randint(1, self._max_num_context+1)

        x_context = np.zeros(shape=(self._batch_size, num_context_points, 1), dtype=np.float32)
        y_context = np.zeros(shape=(self._batch_size, num_context_points, 1), dtype=np.float32)

        # for each batch need to sample some functions from the GP
        for i in range(self._batch_size):

            # if testing want a linspace for xdata - makes plotting nicer
            if self._testing:
                x_data[i, :] = np.reshape(np.linspace(-2, 2, total_points), (-1, 1))
            else:
                x_data[i, :] = np.sort(np.random.uniform(-2, 2, size=(total_points, 1)), axis=0)

            # sample function from GP
            if self._single_kernel:
                kernel = GP.kernels.RBF(length_scale=self._l_scale)
                gp = GP.GaussianProcessRegressor(kernel=kernel)
                y_data[i, :] = gp.sample_y(x_data[i, :], random_state=None)

            else:  # TODO  implement multi-kernel
                pass

            # randomly sample the context vectors
            context_idxs = np.sort(np.random.choice(np.arange(total_points), num_context_points, replace=False))
            x_context[i, :] = x_data[i, context_idxs]
            y_context[i, :] = y_data[i, context_idxs]

        inputs = (x_context, y_context, x_data)
        targets = y_data
        return data_description(Inputs=inputs,
                                Targets=targets)

    def fit_gp(self, inputs):
        """
        Fits a GP to the final sample from the batch to compare with CNP fit
        :param x_context: ndarray size '[batch_size, num_context_points]'
        :param y_context: ndarray size '[batch_size, num_context_points]'
        :param x_data: ndarray size '[batch_size, total_points]'
        :return: y predictions and standard deviations
        """
        x_context, y_context, x_data = inputs
        kernel = GP.kernels.RBF(length_scale=self._l_scale, length_scale_bounds=(1e-2, 1e3))
        gp = GP.GaussianProcessRegressor(kernel=kernel).fit(x_context[-1, :], y_context[-1, :])
        y_prediction, y_prediction_std = gp.predict(x_data[-1, :], return_std=True)
        return y_prediction[np.newaxis, :], y_prediction_std[np.newaxis, :, np.newaxis]

    def plot_fit(self, inputs, targets, y_pred, y_std):
        """
        Plots fit of prediction for final function in batch
        :param x_data: ndarray size '[batch_size, total_points]'
        :param y_data: ndarray size '[batch_size, total_points]'
        :param x_context: ndarray size '[batch_size, num_context_points]'
        :param y_context: ndarray size '[batch_size, num_context_points]'
        :param y_pred: array size '[total_points, 1]'
        :param y_std: array size '[total_points, 1]'
        :return: None
        """
        x_context, y_context, x_data = inputs
        y_data = targets
        x_context = np.squeeze(x_context, axis=-1)
        y_context = np.squeeze(y_context, axis=-1)
        x_data = np.squeeze(x_data, axis=-1)
        y_data = np.squeeze(y_data, axis=-1)
        y_pred = np.squeeze(y_pred, axis=-1)
        y_std = np.squeeze(y_std, axis=-1)

        plt.figure()
        plt.plot(x_data[-1], y_data[-1], 'k--', label='Ground Truth')
        plt.plot(x_context[-1], y_context[-1], 'ko', label='Context')
        plt.plot(x_data[-1], y_pred[-1], 'b-', label='GP Fit')
        plt.fill_between(x_data[-1], y_pred[-1] - 1.96 * y_std[-1],
                         y_pred[-1] + 1.96 * y_std[-1],
                         alpha=0.5, color='b')
        # plt.fill_between(x_data[-1], y_pred[-1] - 1 * y_std[-1],
        #                  y_pred[-1] + 1 * y_std[-1],
        #                  alpha=0.5, color='b')

        # make plots look nice
        plt.xticks(np.arange(-2, 3))
        plt.yticks(np.arange(-2, 3))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    gp_test = GaussianProcess(10,10,testing=False)
    data_test = gp_test.generate_curves()
    inputs, targets = data_test.Inputs, data_test.Targets
    x_context, y_context, x_data = inputs
    y_data = targets
    y_pred, y_std = gp_test.fit_gp(x_context, y_context, x_data)
    gp_test.plot_fit(inputs, targets, y_pred, y_std)



