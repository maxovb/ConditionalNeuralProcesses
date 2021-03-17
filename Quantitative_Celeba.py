import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
from PIL import Image
import cv2 # need to do pip install opencv-python
from sklearn.neighbors import NearestNeighbors
import sklearn.gaussian_process as GP
from Utils.celebaProcessor import process_images
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess


def knn_reconstruction(im,processed,nbr_neighbours):
    knn_reconstructed = np.zeros((1,target_size[0],target_size[1],3)) # pre-allocated memory for the completed image
    knn = NearestNeighbors(n_neighbors=nbr_neighbours).fit(processed.Inputs[0][0])
    # compute the knn reconstruction for all points
    for pts in processed.Inputs[2][0]:
        distances, indices = knn.kneighbors(np.array([pts]))
        neighbours = np.round(processed.Inputs[0][0,indices] * np.expand_dims(np.array([(target_size[0]-1),(target_size[1]-1)]),axis = 0))
        # compute the average over the neighbours
        for u,pos in enumerate(neighbours[0]):
            idx1 = round(pos[0])
            idx2 = round(pos[1])
            if u == 0:
                val = im[0,idx1,idx2,:]/255.
            else:
                val += im[0,idx1,idx2,:]/255.
        val /= (u+1)
        output = val
        knn_reconstructed[0,round(pts[0] * (target_size[0]-1)),round(pts[1] * (target_size[1]-1)),:] = output

    # add the non-masked points
    for l,pts in enumerate(processed.Inputs[0][0]):
        x1 = pts[0]
        x2 = pts[1]
        knn_reconstructed[0,round(x1 * (target_size[0]-1)),round(x2 * (target_size[1]-1)),:] = processed.Inputs[1][0,l,:]
    
    return knn_reconstructed

def gp_reconstruction(processed,target_size):
    gp_reconst = GP.GaussianProcessRegressor(kernel = GP.kernels.RBF(length_scale_bounds=(1e-8, 1)), normalize_y = True)
    gp_reconst.fit(processed.Inputs[0][0], processed.Inputs[1][0])
    gp_reconstructed = gp_reconst.predict(processed.Inputs[2][0])
    gp_reconstructed = np.resize(gp_reconstructed,(1,target_size[0],target_size[0],3))
    return gp_reconstructed


if __name__ == "__main__":     
    test_directory = "DataSets/CelebA/test/"   
    model_types = [[True,False],[True,False],[False,True]] #[CNP,AttentiveCNP,ConvCNP]
    context_ordered = [True,False]
    nbr_context = [10,100,1000]
    results = np.zeros((len(model_types),len(context_ordered),len(nbr_context))) # array to store the results
    comparison = np.zeros((len(context_ordered),len(nbr_context),2)) # KNN and GP
    for i,p in enumerate(model_types):
        # define what to do and type of network
        attention = p[0] # use attention
        convolutional = p[1] # do not set both attention and convolutional to true

        # size of the image
        target_size = (32,32)

        # path to load models
        loading_path = os.path.join(os.getcwd(), "saved_models/CelebA/attention_100kiterations_batch8/")

        # encoder and decoder layer widths (not used for convCNP (define directly in the parameters))
        encoder_layer_widths = [128,128] # not needed for convCNP
        decoder_layer_widths = [64,64,64,64,6] # not needed for convCNP

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

        # load the model
        cnp.load_weights(loading_path)

        nbr_img = 0 # count for the number of images
        for img_name in os.listdir(test_directory):
            nbr_img += 1
            print(nbr_img)
            # open the image and resize
            im = Image.open(test_directory + img_name)
            im = im.resize(target_size, Image.ANTIALIAS)
            im = np.expand_dims(np.array(im),axis = 0).astype('float32')
            for j,use_order in enumerate(context_ordered):
                for k,nbr_points in enumerate(nbr_context):
                    # process the image
                    processed = process_images(im, context_points=nbr_points, convolutional=convolutional, ordered=use_order)
                    means, stds = cnp(processed.Inputs)
                    if not(convolutional):
                        means = np.resize(means,(1,target_size[0],target_size[1],3))
                    results[i,j,k] = (results[i,j,k]*(nbr_img-1) + np.mean((means - im/255.)**2))/nbr_img # running average
                    if i == 0: # run knn and gp only once
                        # knn
                        nbr_neighbours = 1
                        knn_reconstructed = knn_reconstruction(im,processed, nbr_neighbours)
                        comparison[i,k,0] = (comparison[i,k,0]*(nbr_img-1) + np.mean((knn_reconstructed - im/255.)**2))/nbr_img # running average

                        #gp
                        gp_reconstructed = gp_reconstruction(processed,target_size)
                        comparison[i,k,1] = (comparison[i,k,0]*(nbr_img-1) + np.mean((gp_reconstructed - im/255.)**2))/nbr_img # running average

                    print('ACNP','ordered',use_order,'nbr_ctx_points',nbr_points,'sq error:',results[i,j,k])
                    print('KNN','ordered',use_order,'nbr_ctx_points',nbr_points,'sq error:',comparison[i,k,0])
                    print('GP','ordered',use_order,'nbr_ctx_points',nbr_points,'sq error:',comparison[i,k,1])





