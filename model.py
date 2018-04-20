import tensorflow as tf
from attention_blocks import *

def build_model(d_model):
    N = 6 # Number of stacked encoder/decoder layers

    #TODO: Add placeholder or Input layer
    x = tf.placeholder()
    
    #TODO: Add positional encoding

    # Encoder (stacked N times)
    for i in range(N):
        multihead_out = multihead_attention(x,x,x)
        resid1_out = residual_connection(x,multihead_out)
        norm_out = layer_norm(resid1_out)

        ppff_out = positionwise_feedforward(norm_out)
        resid2_out = residual_connection(norm_out,ppff_out)
        x = layer_norm(resid2_out)

    # TODO: Add output (output so far) layer
    y = tf.placeholder()

    # Decoder (stacked N times)
    for i in range(N):
        multihead_out = multihead_attention(y,y,y)
        resid1_out = residual_connection(y,multihead_out)
        norm1_out = layer_norm(resid1_out)

        enc_dec_attn = multihead_attention(norm1_out,x,x) #Use x (from encoder) as key's and values
        resid2_out = resid1_out(norm1_out,enc_dec_attn)
        norm2_out = layer_norm(resid2_out)

        ppff_out = positionwise_feedforward(norm2_out)
        resid3_out = residual_connection(norm2_out,ppff_out)
        y = layer_norm(resid3_out)

    y_transform = tf.layers.dense(y,d_model)
