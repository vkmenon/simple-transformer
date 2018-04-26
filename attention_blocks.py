import tensorflow as tf

#TODO: FIX FOR BATCHING
def attention(query,key,value, mask=False):
    '''
    Returns:
    context => [batch x num_words x dims]
    '''
    score = tf.matmul(query,key,transpose_b=True)
    d_k = query.shape[-1]
    scaled_score = tf.multiply(score, tf.constant(1/d_k.value,dtype=tf.float32))
    if mask:
        pass #TODO: Add mask function
    attn = tf.nn.softmax(scaled_score)
    #TODO: Dropout here
    context = tf.matmul(attn,value)
    
    return context

#TODO: FIX FOR BATCHING
def multihead_attention(query,key,value,mask=False,heads=8):
    '''
    Q, K and V should be of format [Batch x Number of Words x Embedding Dimension]
    '''
    d_model = query.get_shape().as_list()[-1] # Embedding Dimension
    
    assert d_model / heads == 0, 'Embedding dimensions must be divisble by number of heads'

    d_k = d_model / heads

    # Apply a Linear transform to Q, K and V and split up the Q, K and V vectors into separate 'heads'
    # Each element in the following lists has a value of [Batch x Num Words x d_k]
    Q_heads = multihead_linear(query,heads,d_k)
    K_heads = multihead_linear(key,heads,d_k)
    V_heads = multihead_linear(value,heads,d_k)

    # Compute Attention on each head separately
    context_heads = [attention(Q_heads[i],V_heads[i],K_heads[i],mask=mask) for i in range(heads)]

    # Concatenate Attention results
    context = tf.concat(context_heads,axis=-1)

    # Apply a final Linear transform
    context_transform = tf.layers.dense(context,d_model)
    #TODO: Dropout here

    return context_transform

def positionwise_feedforward(x):
    '''
    "Each of the layers in our encoder and decoder contains a fully connected feed-forward network, 
    which is applied to each position separately and identically. This consists of two linear 
    transformations with a ReLU activation in between."

    "While the linear transformations are the same across different positions, they use different
    parameters from layer to layer. Another way of describing this is as two convolutions with 
    kernel size 1. The dimensionality of input and output is dmodel=512, and the inner-layer has 
    dimensionality dff=2048."
    '''
    x = tf.keras.layers.Conv1D(filters=2048,kernel_size=1,activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512,kernel_size=1,activation='linear')(x)

    return x

def multihead_linear(x,num_heads,d_k):
    '''
    Create multiple linear projections of sequential data.

    num_heads = Number of projections of the data
    d_k = output dimension of each projection

    Simple implementation:
    projections = [tf.layers.dense(x,d_k) for i in range(num_heads)]

    #TODO: 
    Implement here as a Conv1D with kernel_size = 1 and depth = num_heads * d_k.
    The output tensor of Conv1D is then split into (num_heads) tensors of depth d*k.
    This allows for a single matmul which is probably more efficient.
    '''
    projections = [tf.layers.dense(x,d_k) for i in range(num_heads)]
    return projections

def layer_norm(x):
    #TODO:
    return x

def residual_connection(input_tensor,output_tensor):
    return tf.keras.layers.add([input_tensor,output_tensor])