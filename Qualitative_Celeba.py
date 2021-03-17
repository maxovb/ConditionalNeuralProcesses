import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from PIL import Image
from Utils.celebaProcessor import process_images, format_context_points_image
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess


if __name__ == "__main__":     
    test_directory = "DataSets/CelebA/test/"   
    context_ordered = [True,False] #[True]
    nbr_context = [1,10,100,500,1000]
    nbr_points_super_res = 512

    # images to test on 
    imgs_name = ["182435.jpg","202562.jpg"]
    imgs_name_super_res = ["182500.jpg","202386.jpg"]

    #directory where to save the output images
    output_dir = "output/CelebA/image_completion/low_resolution/"
    output_dir_super_res = "output/CelebA/image_completion/super_resolution/"

    # size of the image
    target_size = (32,32) #(128,128)
    target_size_small = target_size # for super-resolution
    target_size_medium = (64,64) # for super-resolution
    target_size_large = (128,128)

    # path to load models
    load_CNP = os.path.join(os.getcwd(), "saved_models/CelebA/CNP_200kiterations_batch8/")
    load_attentive = os.path.join(os.getcwd(), "saved_models/CelebA/attention_200kiterations_batch8/")
    load_convolutional = os.path.join(os.getcwd(), "saved_models/CelebA/ConvCNP_200kiterations_batch8/")

    # encoder and decoder layer widths
    encoder_layer_widths_CNP = [128,128,128]
    decoder_layer_widths_CNP = [128,128,128,128,6] 

    encoder_layer_widths_ACNP = [128,128]
    decoder_layer_widths_ACNP = [64,64,64,64,6] 

    # parameters for attention
    attention_params = {"embedding_layer_width":128, "num_heads":8, "num_self_attention_blocks":2}

    # parameters for convolutional
    kernel_size_encoder =9
    kernel_size_decoder = 5 
    convolutional_params = {"number_filters": 128, "kernel_size_encoder":9, "kernel_size_decoder": 5, "number_residual_blocks":4, "convolutions_per_block":1, "output_channels":3}

    # define the models
    cnp_CNP= ConditionalNeuralProcess(encoder_layer_widths_CNP, decoder_layer_widths_CNP, attention = False, attention_params = attention_params, convolutional = False, convolutional_params=convolutional_params)
    cnp_ACNP = ConditionalNeuralProcess(encoder_layer_widths_ACNP, decoder_layer_widths_ACNP, attention = True, attention_params = attention_params, convolutional = False, convolutional_params=convolutional_params)
    cnp_ConvCNP = ConditionalNeuralProcess(encoder_layer_widths_ACNP, decoder_layer_widths_ACNP, attention = False, attention_params = attention_params, convolutional = True, convolutional_params=convolutional_params)

    # load the models
    cnp_CNP.load_weights(load_CNP)
    cnp_ACNP.load_weights(load_attentive)
    cnp_ConvCNP.load_weights(load_convolutional)

    # store the models in a list
    models = [cnp_CNP,cnp_ACNP,cnp_ConvCNP]
    model_names = ["CNP","ACNP","ConvCNP"]

    # image inpainting with ordered/non-ordered with different number of ctxt pixels
    for ordered in context_ordered:
        for nbr_points in nbr_context:
            for i,img_name in enumerate(imgs_name):
                # load the iamge
                im_dir = test_directory + img_name
                im = Image.open(im_dir)
                im_original = np.array(im).astype('float32')
                im = im.resize(target_size, Image.ANTIALIAS)
                im = np.expand_dims(np.array(im),axis = 0).astype('float32')

                # prepare the data
                processed = process_images(im, context_points=nbr_points, convolutional=False, ordered=ordered)
                processed_convolutional = process_images(im, context_points=nbr_points, convolutional=True, ordered=ordered, pre_defined = True, pre_defined_values = processed.Inputs[0]) 

                # save the original image
                im_out= Image.fromarray(im_original.astype(np.uint8))
                im_out.save(output_dir + "img" + str(i) + "_true.png")
                
                # save the original image low resolution
                im_out = Image.fromarray(im.reshape(target_size[0], target_size[1], 3).astype(np.uint8))
                im_out.save(output_dir + "img" + str(i) + "_true_low_res.png")

                # save the image only with the context pixels
                context_image = format_context_points_image(processed.Inputs)
                im_out = Image.fromarray((context_image).reshape(target_size[0], target_size[1], 3).astype(np.uint8))
                if ordered:
                    im_out.save(output_dir + "img" + str(i) + "_ordered__context" + str(nbr_points) + ".png")
                else:
                    im_out.save(output_dir + "img" + str(i) + "_context" + str(nbr_points) + ".png")

                for j,model in enumerate(models):
                    if j < 2: # if CNP or ACNP
                        means, stds = model(processed.Inputs)
                    else: # if ConvCNP
                        means, stds = model(processed_convolutional.Inputs)
                        means = np.resize(means,(1,target_size[0],target_size[1],3))
                        stds = np.resize(stds,(1,target_size[0],target_size[1],3))

                    if ordered:
                        attrib = "_" + model_names[j] + "_ordered" + "_context" + str(nbr_points)
                    else:
                        attrib = "_" + model_names[j] + "_context" + str(nbr_points) 

                    # save the mean
                    im_out = Image.fromarray(np.array(means*255).reshape(target_size[0], target_size[1], 3).astype(np.uint8))
                    im_out.save(output_dir + "img" + str(i) + "_mean" + attrib + ".png")
            
                    # save the image only with the context pixels
                    im_out = Image.fromarray(np.array(stds*255).reshape(target_size[0], target_size[1], 3).astype(np.uint8))
                    im_out.save(output_dir + "img" + str(i) + "_std" + attrib + ".png")

    # different resolution image inpainting
    for i,img_name in enumerate(imgs_name_super_res):
        # load the iamge
                im_dir = test_directory + img_name
                im = Image.open(im_dir)
                im_full = im.resize(target_size_large, Image.ANTIALIAS)
                im_medium = im.resize(target_size_medium, Image.ANTIALIAS)
                im_small = im.resize(target_size_small, Image.ANTIALIAS)
                im_full = np.expand_dims(np.array(im_full),axis = 0).astype('float32')
                im_medium = np.expand_dims(np.array(im_medium),axis = 0).astype('float32')
                im_small = np.expand_dims(np.array(im_small),axis = 0).astype('float32')

                # prepare the data
                ordered = False
                processed_small = process_images(im_small, context_points=nbr_points_super_res, convolutional=False, ordered=ordered)
                processed_small_convolutional = process_images(im_small, context_points=nbr_points_super_res, convolutional=True, ordered=ordered, pre_defined = True, pre_defined_values = processed_small.Inputs[0]) 
                processed_medium = process_images(im_medium, context_points=nbr_points_super_res, convolutional=False, ordered=ordered, pre_defined_values = processed_small.Inputs[0]) 
                processed_medium_convolutional = process_images(im_medium, context_points=nbr_points_super_res, convolutional=True, ordered=ordered, pre_defined = True, pre_defined_values = processed_small.Inputs[0])
                processed_large = process_images(im_full, context_points=nbr_points_super_res, convolutional=False, ordered=ordered, pre_defined_values = processed_small.Inputs[0]) 
                processed_large_convolutional = process_images(im_full, context_points=nbr_points_super_res, convolutional=True, ordered=ordered, pre_defined = True, pre_defined_values = processed_small.Inputs[0]) 

                # save the original image
                im_out = Image.fromarray((processed_large.Targets*255).reshape(target_size_large[0], target_size_large[1], 3).astype(np.uint8))
                im_out.save(output_dir_super_res + "img" + str(i) + "_true.png")

                # save the original image lower resolutions
                # small
                im_out = Image.fromarray((im_small).reshape(target_size_small[0], target_size_small[1], 3).astype(np.uint8))
                im_out.save(output_dir_super_res + "img" + str(i) + "_true_small.png")
                # medium
                im_out = Image.fromarray((im_medium).reshape(target_size_medium[0], target_size_medium[1], 3).astype(np.uint8))
                im_out.save(output_dir_super_res + "img" + str(i) + "_true_medium.png")
        
                # save the image only with the context pixels
                context_image = format_context_points_image(processed_small.Inputs)
                im_out = Image.fromarray(context_image.astype(np.uint8))
                im_out.save(output_dir_super_res + "img" + str(i) + "_context.png")

                for j,model in enumerate(models):
                    if j < 2: # if CNP or ACNP
                        means_small, stds_small = model(processed_small.Inputs)
                        means_medium, stds_medium = model(processed_medium.Inputs)
                        means_large, stds_large = model(processed_large.Inputs)

                    else: # if ConvCNP
                        means_small, stds_small = model(processed_small_convolutional.Inputs)
                        means_medium, stds_medium = model(processed_medium_convolutional.Inputs)
                        means_large, stds_large = model(processed_large_convolutional.Inputs)

                    # small
                    im_out = Image.fromarray((np.array(means_small)*255).reshape(target_size_small[0], target_size_small[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_mean_small_" + model_names[j] + ".png")

                    im_out = Image.fromarray((np.array(stds_small)*255).reshape(target_size_small[0], target_size_small[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_std_small_" + model_names[j] + ".png")

                    # medium
                    im_out = Image.fromarray((np.array(means_medium)*255).reshape(target_size_medium[0], target_size_medium[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_mean_medium_" + model_names[j] + ".png")

                    im_out = Image.fromarray((np.array(stds_medium)*255).reshape(target_size_medium[0], target_size_medium[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_std_medium_" + model_names[j] + ".png")

                    # medium
                    im_out = Image.fromarray((np.array(means_large)*255).reshape(target_size_large[0], target_size_large[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_mean_large_" + model_names[j] + ".png")

                    im_out = Image.fromarray((np.array(stds_large)*255).reshape(target_size_large[0], target_size_large[1], 3).astype(np.uint8))
                    im_out.save(output_dir_super_res + "img" + str(i) + "_std_large_" + model_names[j] + ".png")

    print('finished !')

