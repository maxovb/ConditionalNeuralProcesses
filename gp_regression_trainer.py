import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GaussianProcesses.GaussianProcessSampler import GaussianProcess
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf
import os
import time
from datetime import datetime
from tqdm import tqdm


def train(cnp, batch_size=64, max_iters=100000):
    # tf.config.run_functions_eagerly(True)
    percentage = 1
    default_max_context = np.random.randint(2, percentage * 50)
    gp_train = GaussianProcess(batch_size, default_max_context, testing=False)
    loss = []
    start = time.perf_counter()
    for i in tqdm(range(1, max_iters)):

        try:
            data_train = generate_gp_samples(gp_train)
        except:
            continue

        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        # every 1000 iterations try new max contexts with big batch size to avoid overfitting
        if i % 1000 == 0:
            # data_val = generate_gp_samples(gp_train, gen_new_gp=True)
            # val_loss = cnp.train_step(data_val.Inputs, data_val.Targets)
            print(f"Running avg (1000) loss at iteration {i} is: {np.mean(loss[-1000:])}")
            # print(f"Validation Loss at iteration {i} is: {val_loss}")

        # # early stopping
        # if i > 2000:
        #     # check any progress actually being made
        #     # (just to make computationally less expensive)
        #     if np.mean(loss[-2000:-1000]) - np.mean(loss[-1000:]) < 0:
        #         break
    end = time.perf_counter()
    return cnp, loss, end - start


def generate_gp_samples(gp_object, gen_new_gp=False):
    if gen_new_gp:
        max_context = np.random.choice([10, 20, 50], replace=True)
        gp_object = GaussianProcess(2048, max_context, testing=False)
    return gp_object.generate_curves()


if __name__ == "__main__":
    load = True
    save = False
    training = False
    Testing = True
    attention = True # use attention
    loading_path = os.path.join(os.getcwd(), "saved_models/GP_Regression/attention_100kiterations_batch64/")
    saving_path = os.path.join(os.getcwd(), "saved_models/long_colab_run/")
    encoder_layer_widths = [128,128]
    decoder_layer_widths = [64,64,64,64,2]
    attention_params = {"embedding_layer_width":128, "num_heads":8, "num_self_attention_blocks":2}
    cnp = ConditionalNeuralProcess(encoder_layer_widths, decoder_layer_widths, attention, attention_params)
    
    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, batch_size=64)
        print(total_runtime)
        avg_loss = pd.Series(loss).rolling(window=1000).mean().iloc[1000 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.show()
    if save:
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        cnp.save_weights("saved_models/GP_Regression/" + current_time + "/", overwrite=False)
        # cnp.save_weights(saving_path)
    if Testing:
        gp = GaussianProcess(1, 7, testing=True)
        data = gp.generate_curves()
        means, stds = cnp(data.Inputs)
        gp_mean, gp_stds = gp.fit_gp(data.Inputs)
        gp.plot_fit(data.Inputs, data.Targets, means.numpy(), stds.numpy())
        gp.plot_fit(data.Inputs, data.Targets, gp_mean, gp_stds)
