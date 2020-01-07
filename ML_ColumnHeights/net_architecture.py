
from keras.models import Model
from net_layers import activation_function,regularization,convolutional_layer,skip_connection,residual_block,convolution_block,max_pooling_layer,bilinear_upsampling,BilinearUpSampling2D,upsampling_layer,score_layer


def CNN(input_tensor,output_channels=1, channels=32):
    
    # encoder : the convolution takes action
    
    # block 1
    down_sampling_block_1 = convolution_block(input_tensor, channels, name="down_sampling_block_1")
    pooling_1 = max_pooling_layer(down_sampling_block_1, name="max_pooling_1")

    # block 2
    down_sampling_block_2 = convolution_block(pooling_1, channels*2, name="down_sampling_block_2")
    pooling_2 = max_pooling_layer(down_sampling_block_2, name="max_pooling_2")

    # block 3
    down_sampling_block_3 = convolution_block(pooling_2, channels*4, name="down_sampling_block_3")
    pooling_3 = max_pooling_layer(down_sampling_block_3, name="max_pooling_3")
    
    # bridge layer between the encoder and the decoder
    bridge_layer = convolution_block(pooling_3, channels*8, name="bridge_layer")
    
    # decoder : the deconvolution takes action
    
    # simmetry with block 3
    up_sampling_block_3 = upsampling_layer(bridge_layer, channels*4, name="upsampling_3_in")
    up_sampling_block_3_connected = skip_connection(up_sampling_block_3, down_sampling_block_3, name='skip_connection_3')
    up_sampling_block_3_convolution = convolution_block(up_sampling_block_3_connected, channels*4, name="up_sampling_3_out")
    
    # simmetry with block 2
    up_sampling_block_2 = upsampling_layer(up_sampling_block_3_convolution, channels*2, name="upsampling_2_in")
    up_sampling_block_2_connected = skip_connection(up_sampling_block_2, down_sampling_block_2, name='skip_connection_2')
    up_sampling_block_2_convolution = convolution_block(up_sampling_block_2_connected, channels*2, name="up_sampling_2_out")
    
    # simmetry with block 1
    up_sampling_block_1 = upsampling_layer(up_sampling_block_2_convolution, channels, name="upsampling_1_in")
    up_sampling_block_1_connected = skip_connection(up_sampling_block_1, down_sampling_block_1, name='skip_connection_1')
    up_sampling_block_1_convolution = convolution_block(up_sampling_block_1_connected, channels, name="up_sampling_1_out")

    # final prediction
    inference = score_layer(up_sampling_block_1_convolution, channels=output_channels)
    return Model(input_tensor, inference)
