import numpy as np
import tensorflow as tf
import collections

data_description = collections.namedtuple("data_description", ("Inputs",
                                                               "Targets"))


def process_images(image_set, context_points=100, convolutional = False, ordered=False):
    """
    :param image_set: Set of images needing processing
    :return: Named tuple containing inputs and targets.
             The x_data is a list of coordinate pairs [x1, x2].
             The y_data is a list of pixel intensities [yT]
    """
    # grab shapes
    batch_size = image_set.shape[0]
    pixel_width = image_set.shape[1]

    # normalise pixel values
    image_set = image_set / 255.0


    if not(convolutional):

        # flatten image into vector (this is the full y data)
        image_set = np.reshape(image_set, (-1, pixel_width**2, 1))

        # create shell containing indices (this is the full x data)
        # first make mask containing indices
        image_indices = np.zeros((pixel_width, pixel_width, 2), dtype=np.int32)

        for i in range(pixel_width):
            for j in range(pixel_width):
                image_indices[i, j] = np.array([i, j])

        image_indices = image_indices.reshape(-1, 2)
        image_set_idxs = np.repeat(image_indices[np.newaxis, :], batch_size, axis=0)

        context_indices = np.zeros((batch_size, context_points), dtype=np.int32)

        # choose context points
        for i in range(batch_size):
            if not(ordered):
                context_indices[i, :] = np.random.choice(np.arange(pixel_width**2,
                                                                dtype=np.int32), size=context_points, replace=False)
            else:
                context_indices[i, :] = np.arange(context_points,dtype=np.int32)


        x_context = image_set_idxs[np.arange(batch_size).reshape(-1, 1), context_indices]

        y_context = image_set[np.arange(batch_size).reshape(-1, 1), context_indices]

        # normalise all vectors so values in [0,1]
        x_context = x_context / (pixel_width - 1)
        image_set_idxs = image_set_idxs / (pixel_width - 1)

        inputs = (x_context.astype("float32"), y_context, image_set_idxs.astype("float32"))
        targets = image_set
    
    else:
        # create the mask
        mask = np.zeros((batch_size, pixel_width, pixel_width,1), dtype=np.int32)
        
        # number of lines to set to 1 before shuffling to get the mask, and number of columns in last line
        nbr_lines = int(context_points/pixel_width)
        nbr_columns_last_lines = context_points % pixel_width
        
        # make the mask by shuffling
        mask[:,:nbr_lines,:] = 1
        mask[:,nbr_lines,:nbr_columns_last_lines] = 1
        if not(ordered):
            for batch_number in range(batch_size):
                flat = np.reshape(mask[batch_number, :, :], (pixel_width*pixel_width,1))
                np.random.shuffle(flat)
                mask[batch_number, :, :] = np.reshape(flat, (pixel_width, pixel_width,1))

        # get the context pixels by masking with the mask
        image_set = np.expand_dims(image_set,axis = -1)
        image_context =  mask *  image_set

        inputs = (mask.astype('float32'), image_context.astype('float32'))

        targets = image_set

    return data_description(Inputs=inputs,
                            Targets=targets)


def sample_context_points(arr, pixel_width, context_points):
    return np.random.choice(np.arange(pixel_width**2, dtype=np.int32), size=context_points, replace=False)


def format_context_points_image(inputs):
    x_context, y_context = inputs[0], inputs[1]

    x_context = (x_context * 27).astype(np.int32)
    y_context = (y_context * 255).astype(np.int32)

    image = np.zeros((28, 28, 3), dtype=np.int32)
    image[:, :, 2] = 255

    for x, y in zip(x_context[0], y_context[0]):
        if y == 0:
            image[x[0], x[1]] = [0, 0, 0]
        else:
            image[x[0], x[1]] = [y, y, y]

    return image


if __name__ == '__main__':
    (train_set, _), (test_set, _) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    data_set = process_images(test_set[:64])
    #test_pic = process_images(test_set[0].reshape(1,28,28))
