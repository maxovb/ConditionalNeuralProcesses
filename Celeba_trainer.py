import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf
import os
import time
from datetime import datetime
from Utils.celebaProcessor import process_images, format_context_points_image
from tqdm import tqdm
import pickle as pckl


def data_generator(directory, type_data, batch_size=64, target_size=(28, 28)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = datagen.flow_from_directory(directory, batch_size=batch_size,
                                            target_size=target_size, shuffle=True, classes=[type_data])
    while True:
        batch = next(generator)
        yield batch


def train(cnp, train_data, batch_size=64, max_iters=50000, convolutional=False):
    """
    Train with batches of size 30 since 60000 images total, 4000 iterations
    Randomly sample number of context  points
    """
    #tf.config.run_functions_eagerly(True)
    loss = []
    start = time.perf_counter()

    for i in tqdm(range(1, max_iters+1)):
        # generate a batch
        batch = next(train_data)[0]
        img_shape = np.array(batch).shape[1:]
        num_context = np.random.randint(2, img_shape[0]**2)
        data_train = process_images(batch, context_points=num_context, convolutional=convolutional)
        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 1000 == 0:
            print(f'The running avg loss at iteration {i} is: {np.mean(loss[-1000:])}')

    end = time.perf_counter()
    return cnp, loss, end-start


def test_cnp(cnp, test_data, context_ratio=0.2, convolutional = False):
    # grab a random image from the test set
    batch = next(test_data)[0]
    img_shape = np.array(batch).shape[1:]

    # process image
    percentage = 0.2
    processed = process_images(batch, context_points=int(percentage*img_shape[0]**2), convolutional=convolutional)

    # evaluate cnp on image
    means, stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, img_shape)
    predictive_stds = tf.reshape(stds, img_shape)
    if not(convolutional):
        context_image = format_context_points_image(processed.Inputs)
    else:
        context_image = processed.Inputs[1][0]

    print(context_image.shape)
    # plot stuff
    plt.figure('context')
    plt.imshow(context_image, cmap='gray')
    plt.title('Context')
    plt.tight_layout()
    plt.show()
    plt.savefig('output/CelebA/context.png')

    # plot stuff
    plt.figure('means')
    plt.imshow(predictive_mean, cmap='gray')
    plt.title('Predictive Mean')
    plt.tight_layout()
    plt.show()
    plt.savefig('output/CelebA/means.png')

    plt.figure('stds') 
    plt.imshow(predictive_stds, cmap='gray')
    plt.title('Predictive Std')
    plt.tight_layout()
    plt.show()
    plt.savefig('output/CelebA/std.png')

    plt.figure('actual')
    plt.imshow(processed.Targets.reshape(img_shape[0], img_shape[1], img_shape[2]), cmap='gray')
    plt.title('Actual')
    plt.tight_layout()
    plt.show()
    plt.savefig('output/CelebA/actual.png')


if __name__ == "__main__":
    # define what to do and type of network
    load = True
    save = False
    training = True
    test = False
    attention = False # use attention
    convolutional = False # do not set both attention and convolutional to true

    # size of the image
    target_size = (32,32)

    # path to load models
    loading_path = os.path.join(os.getcwd(), "saved_models/CelebA/CNP_200kiterations_batch8/")
    saving_path = os.path.join(os.getcwd(), "saved_models/CelebA/")

    # encoder and decoder layer widths (not used for convCNP (define directly in the parameters))
    encoder_layer_widths = [128,128,128] #[128,128] for ACNP  # [128,128,128] for CNP # not needed for convCNP
    decoder_layer_widths = [128,128,128,128,6] #[64,64,64,64,6] for ACNP # [128,128,128,128,6] for CNP # not needed for convCNP

    # parameters for attention
    attention_params = {"embedding_layer_width":128, "num_heads":8, "num_self_attention_blocks":2}

    # parameters for convolutional
    if target_size[0] < 50 and target_size[1] < 50:
        kernel_size_encoder =9
        kernel_size_decoder = 5 
    else:
        kernel_size_encoder = 7
        kernel_size_decoder = 3
    convolutional_params = {"number_filters": 128, "kernel_size_encoder":9, "kernel_size_decoder": 5, "number_residual_blocks":4, "convolutions_per_block":1, "output_channels":3}

    # define the model
    cnp = ConditionalNeuralProcess(encoder_layer_widths, decoder_layer_widths, attention, attention_params, convolutional = convolutional, convolutional_params=convolutional_params)

    # make a generator for the data
    train_data = data_generator('DataSets/CelebA', 'train', batch_size=8, target_size=target_size)
    test_data = data_generator('DataSets/CelebA', 'test', batch_size=1, target_size=target_size)

    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, train_data, max_iters=200000, convolutional=convolutional)
        print(total_runtime)
        avg_loss = pd.Series(loss).rolling(window=100).mean().iloc[100 - 1:].values
        with open('output/CelebA/training_loss/loss_ConvCNP.txt', 'w') as file:
            for listitem in loss:
                file.write('%s\n' % listitem)
        avg_loss = pd.Series(loss).rolling(window=1000).mean().iloc[1000 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.savefig('output/CelebA/training_loss/loss_ConvCNP.eps')

    if save:
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        cnp.save_weights("saved_models/CelebA/" + current_time + "/", overwrite=False)
    if test:
        test_cnp(cnp, test_data, convolutional=convolutional)


    #cnp.model().summary()
