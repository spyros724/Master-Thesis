import tensorflow as tf
from attention import *
from pos_encodings import *

def shallow_mlp_model(in_size, hl_num, hl_size, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(inputs)
    for _ in range(hl_num - 1):
        x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)

def deep_residual_mlp_model(in_size, hl_size, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))

    x_b1 = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(inputs)

    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x_b1)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)

    x_b2 = tf.keras.layers.Add()([x_b1,x])

    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x_b2)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)

    x_b3 = tf.keras.layers.Add()([x_b2,x])

    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x_b3)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


def basic_attention_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)

    attention_output = MultiHeadAttention(num_heads, 36, 36, dim)(x,x,x)
    flattened = tf.keras.layers.Flatten()(attention_output)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)



def mlp_attention_model(in_size, att_dim, num_heads, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    print(1,shape(inputs))
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    print(2,shape(x))
    x = tf.keras.layers.Reshape((in_size, 1))(x)
    print(3,shape(x))
    
    x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x,x,x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)

def positional_attention_model(in_size, att_dim, dim, num_heads, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    inputs = tf.keras.layers.Reshape((in_size, 1))(inputs)
    pos_enc = positional_encoding(in_size, dim)
    x = inputs + pos_enc

    x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x,x,x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)


def mlp_positional_attention_model(in_size, att_dim, dim, num_heads, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    inputs = tf.keras.layers.Reshape((in_size, 1))(x)
    pos_enc = positional_encoding(in_size, dim)
    x = inputs + pos_enc

    x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x,x,x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)

def positional_mlp_attention_model(in_size, att_dim, dim, num_heads, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)

    pos_enc = positional_encoding(in_size, dim)
    pos = x + pos_enc
    pos = tf.reshape(pos, shape=[-1, in_size])
    
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(pos)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Reshape((in_size, 1))(x)

    x = MultiHeadAttention(num_heads, in_size, in_size, att_dim)(x,x,x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=lout)


def lstm_attention_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)
    x = tf.keras.layers.LSTM(36, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Reshape((in_size*36, 1))(x)

    attention_output = MultiHeadAttention(num_heads, 36, 36, dim)(x,x,x)
    flattened = tf.keras.layers.Flatten()(attention_output)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)


def mlp_lstm_attention_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(in_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Reshape((in_size, 1))(x)

    x = tf.keras.layers.LSTM(36, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Reshape((in_size*36, 1))(x)

    attention_output = MultiHeadAttention(num_heads, 36, 36, dim)(x,x,x)
    flattened = tf.keras.layers.Flatten()(attention_output)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=lout)




def attention_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)
    print(shape(x))
    attention_output = MultiHeadAttention(num_heads, 36, 36, dim)(x,x,x)
    flattened = tf.keras.layers.Flatten()(attention_output)
    x_b1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b1)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)

    x_b2 = tf.keras.layers.Add()([x_b1,x])

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b2)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)

    x_b3 = tf.keras.layers.Add()([x_b2,x])

    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x_b3)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)





def additive_lstm_attention_model(in_size, num_heads, dim, lstm_units, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)
    return x


def positional_attention_model(in_size, att_dim, dim, num_heads, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    inputs = tf.keras.layers.Reshape((in_size, 1))(inputs)
    pos_enc = positional_encoding(in_size, dim)
    x = inputs + pos_enc
    print(shape(x))
    x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x,x,x)
    x = tf.keras.layers.Flatten()(x)
    x1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x1)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x2 = tf.keras.layers.Add()([x1,x])
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x2)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    print(shape(lout))
    model = tf.keras.models.Model(inputs=inputs, outputs=lout)
    
    return model


def transformer_model(in_size, att_dim, dim, num_heads, fh):
    #Inputs
    inputs = tf.keras.layers.Input(shape=(in_size,))
    inputs = tf.keras.layers.Reshape((in_size, 1))(inputs)
    
    #Encoder
    pos_enc = positional_encoding(in_size, dim)
    x_embedded = inputs + pos_enc
    x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x_embedded,x_embedded,x_embedded)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x_add = tf.keras.layers.Add()([x_embedded,x])
    x = tf.keras.layers.Flatten()(x_add)
    x1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x1)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x2 = tf.keras.layers.Add()([x1,x])
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x2)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=lout)
    
    return model



########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################













# def transformer(in_size, att_dim, dim, num_heads, layers,fh):
#     # Input layer
#     inputs = tf.keras.layers.Input(shape=(in_size,))
    
#     # Positional encoding
#     print(in_size)
#     positional_encoding_layer = PositionalEmbedding(max_seq_len=in_size, d_model=dim)
#     x = positional_encoding_layer(inputs)
    
#     # Encoder
#     for _ in range(layers):
#         x = tf.keras.layers.Reshape((in_size, 1))(x)
#         x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, name="decoder_attention")(queries=x, keys=x, values=x) #x = MultiHeadAttention(num_heads, 36, 36, att_dim)(x,x,x)      #tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)([x, x])
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         x = tf.keras.layers.Dropout(0.1)(x)
#         x_ff = tf.keras.layers.Dense(256, activation='relu')(x)
#         x_ff = tf.keras.layers.Dense(att_dim)(x_ff)
#         x = tf.keras.layers.Add()([x, x_ff])
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         new_shape = (att_dim*dim*dim // 1, 1)
#         x= tf.reshape(x, new_shape)
#         print(tf.shape(x))
    
#     # Decoder
#     for _ in range(layers):
        
#         x = MultiHeadAttention(num_heads, att_dim*dim*dim, att_dim*dim*dim, att_dim)(x,x,x)
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         x = tf.keras.layers.Dropout(0.1)(x)
#         x_ff = tf.keras.layers.Dense(256, activation='relu')(x)
#         x_ff = tf.keras.layers.Dense(att_dim)(x_ff)
#         x = tf.keras.layers.Add()([x, x_ff])
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
#     # Output layer
#     lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
#     model = tf.keras.models.Model(inputs=inputs, outputs=lout)
    
#     return model









# def mlp_attention_model(in_size,hl_num, hl_size, num_heads, dim, fh):
#     inputs = tf.keras.Input(shape=(in_size,))
#     x = tf.keras.layers.Reshape((in_size, 1))(inputs)
#     x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
#     for _ in range(hl_num - 1):
#         x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
#     attention_output = MultiHeadAttention(num_heads, dim, dim, num_heads * dim)(x,x,x)
#     flattened = tf.keras.layers.Flatten()(attention_output)
#     x = tf.keras.layers.Dense(3*fh, activation='relu', kernel_initializer='he_uniform')(flattened)
#     lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
#     return tf.keras.models.Model(inputs=inputs, outputs=lout)



def lstm_attention_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)
    x = tf.keras.layers.LSTM(6, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Reshape((in_size*6, 1))(x)
    attention_output = MultiHeadAttention(num_heads, 36, 36, dim)(x,x,x)
    flattened = tf.keras.layers.Flatten()(attention_output)
    x_b1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)
    #x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b1)
    #x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    #x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    #x_b2 = tf.keras.layers.Add()([x_b1,x])
    #x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b2)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b1)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    #x_b3 = tf.keras.layers.Add()([x_b2,x])
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)

# def lstm_attention_model(in_size, num_heads, dim, fh):
#     inputs = tf.keras.Input(shape=(in_size,))
#     x = tf.keras.layers.Reshape((in_size, 1))(inputs)
#     x = tf.keras.layers.LSTM(6, return_sequences=True)(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Reshape((in_size*6, 1))(x)
    
#     # Use MultiHeadSelfAttention for self-attention
#     attention_output = tf.keras.layers.MultiHeadAttention(
#         num_heads=num_heads,
#         key_dim=dim,
#         value_dim=dim,
#     )(x, x, x)
    
#     flattened = tf.keras.layers.Flatten()(attention_output)
#     x_b1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(flattened)
#     x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_b1)
#     x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
#     lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
#     return tf.keras.models.Model(inputs=inputs, outputs=lout)



def attention_lstm_model(in_size, num_heads, dim, fh):
    inputs = tf.keras.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)
    attention_output = MultiHeadAttention(num_heads, dim, dim, num_heads * dim)(x,x,x)
    x = tf.keras.layers.LSTM(fh, return_sequences=True)(attention_output)
    flattened = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3*fh, activation='relu', kernel_initializer='he_uniform')(flattened)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)































































def deep_residual_mlp_model_exog(in_size, exog_sizes, hl_size, fh):
    # past observations flow
    inputs_ts = tf.keras.layers.Input(shape=(in_size,))
    x_b1 = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(inputs_ts)

    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x_b1)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)

    x_b2 = tf.keras.layers.Add()([x_b1,x])

    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x_b2)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)

    x_b3 = tf.keras.layers.Add()([x_b2,x])

    # x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x_b3)
    # x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    # x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)

    # x_b4 = tf.keras.layers.Add()([x_b3,x])

    ts_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x_b3)

    exog_outs = [ts_out]
    exog_inputs = [inputs_ts]
    if len(exog_sizes)==1:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)
    elif len(exog_sizes)==2:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)
    else:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)

        inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog3)

        # inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        # exog3 = tf.keras.layers.Dense((2*exog_sizes[2]), activation='linear', kernel_initializer='he_uniform')(inputs_exog3)
        # exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog3)

        exog_outs.append(exog3_out)
        exog_inputs.append(inputs_exog3)

    # Merge flows
    x_add = tf.keras.layers.Add()(exog_outs)

    return tf.keras.models.Model(inputs=exog_inputs, outputs=x_add)

def basic_mlp_model_exog(in_size, exog_sizes, hl_size, fh):
    # past observations flow
    inputs_ts = tf.keras.layers.Input(shape=(in_size,))
    
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(inputs_ts)
    ts_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)

    exog_outs = [ts_out]
    exog_inputs = [inputs_ts]
    if len(exog_sizes)==1:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)
    elif len(exog_sizes)==2:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)
    else:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)

        inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog3)

        # inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        # exog3 = tf.keras.layers.Dense((2*exog_sizes[2]), activation='linear', kernel_initializer='he_uniform')(inputs_exog3)
        # exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog3)

        exog_outs.append(exog3_out)
        exog_inputs.append(inputs_exog3)

    # Merge flows
    x_add = tf.keras.layers.Add()(exog_outs)

    return tf.keras.models.Model(inputs=exog_inputs, outputs=x_add)

def shallow_mlp_model_exog(in_size, exog_sizes, hl_size, fh):
    # past observations flow
    inputs_ts = tf.keras.layers.Input(shape=(in_size,))
    
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(inputs_ts)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(hl_size, activation='relu', kernel_initializer='he_uniform')(x)
    ts_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)

    exog_outs = [ts_out]
    exog_inputs = [inputs_ts]
    if len(exog_sizes)==1:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)
    elif len(exog_sizes)==2:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)
    else:
        inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        
        # inputs_exog1 = tf.keras.layers.Input(shape=(exog_sizes[0],))
        # exog1 = tf.keras.layers.Dense((2*exog_sizes[0]), activation='linear', kernel_initializer='he_uniform')(inputs_exog1)
        # exog1_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog1)

        exog_outs.append(exog1_out)
        exog_inputs.append(inputs_exog1)

        inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog2)

        # inputs_exog2 = tf.keras.layers.Input(shape=(exog_sizes[1],))
        # exog2 = tf.keras.layers.Dense((2*exog_sizes[1]), activation='linear', kernel_initializer='he_uniform')(inputs_exog2)
        # exog2_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog2)

        exog_outs.append(exog2_out)
        exog_inputs.append(inputs_exog2)

        inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(inputs_exog3)

        # inputs_exog3 = tf.keras.layers.Input(shape=(exog_sizes[2],))
        # exog3 = tf.keras.layers.Dense((2*exog_sizes[2]), activation='linear', kernel_initializer='he_uniform')(inputs_exog3)
        # exog3_out = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(exog3)

        exog_outs.append(exog3_out)
        exog_inputs.append(inputs_exog3)

    # Merge flows
    x_add = tf.keras.layers.Add()(exog_outs)

    return tf.keras.models.Model(inputs=exog_inputs, outputs=x_add)


def deep_residual_bilstm_model(in_size, n_cells, fh):
    inputs = tf.keras.layers.Input(shape=(in_size,))
    x = tf.keras.layers.Reshape((in_size, 1))(inputs)

    x_b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x_b1)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x)

    x_b2 = tf.keras.layers.Add()([x_b1,x])


    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x_b2)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x)

    x_b3 = tf.keras.layers.Add()([x_b2,x])


    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x_b3)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_cells, activation='relu', return_sequences=True))(x)

    x_b4 = tf.keras.layers.Add()([x_b3,x])
   
    x = tf.keras.layers.Flatten()(x_b4)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    lout = tf.keras.layers.Dense(fh, activation='linear', kernel_initializer='he_uniform')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=lout)
