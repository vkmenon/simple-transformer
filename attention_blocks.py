import tensorflow as tf

def attention(query,key,value, mask=False):
    '''
    Returns:
    context => [batch x num_words x dims]
    '''
    score = tf.matmul(query,key,transpose_b=True)
    d_k = query.shape[-1]
    scaled_score = score / d_k
    if mask:
        pass #TODO: Add mask function
    attn = tf.nn.softmax(scaled_score)
    context = tf.matmul(attn,value)
    
    return context

def multihead_attention(query,key,value,heads=8):
    '''
    Q, K and V should be of format [Batch x Number of Words x Embedding Dimension]
    '''

    d_model = query.shape[-1] # Embedding Dimension

    # Apply a Linear transform to Q, K and V
    Q = tf.layers.dense(query,d_model)
    K = tf.layers.dense(key,d_model)
    V = tf.layers.dense(value,d_model)

    # Split up the Q, K and V vectors into separate 'heads'
    # Each element in the following lists has a value of [Batch x Num Words x d_model/heads]
    Q_heads = tf.split(axis=-1,value=Q, num_split=heads)
    V_heads = tf.split(axis=-1,value=K, num_split=heads)
    K_heads = tf.split(axis=-1,value=V, num_split=heads)

    # Compute Attention on each head separately
    context_heads = [attention(Q_heads[i],V_heads[i],K_heads[i]) for i in range(heads)]

    # Concatenate Attention results
    context = tf.concat(context_heads,axis=-1)

    # Apply a final Linear transform
    context_transform = tf.layers.dense(context,d_model)

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

def layer_norm(x):
    return x

def residual_connection(input_tensor,output_tensor):
    return tf.keras.layers.add([input_tensor,output_tensor])