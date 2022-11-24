from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import pad, maximum, random, int32
import keras.activations

#Original WaveGAN: https://github.com/chrisdonahue/wavegan
#Label embeding using the method in https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

def generator(z_dim = 100, use_batch_norm = True, n_classes = 5):
    generator_filters = [512, 256, 128, 64, 1]

    label_input = Input(shape=(1,), dtype='int32', name='generator_label_input')
    label_em = Embedding(n_classes, n_classes * 10, name = 'label_embedding')(label_input)
    label_em = Dense(32, name = 'label_dense')(label_em)
    label_em = Reshape((8, 4, 1), name = 'label_reshape')(label_em)
    
    generator_input = Input(shape=(z_dim,), name='generator_input')
    x = generator_input
    x = Dense(16384, name='generator_input_dense')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((8, 4, 512), name='generator_input_reshape')(x)
    # if use_batch_norm == True:
    #     x = BatchNormalization()(x)
    
    x = Concatenate()([x, label_em]) 
    
    for i in range(4):
        print(generator_filters[i])
        x = Conv2DTranspose(filters=generator_filters[i], kernel_size=(12, 3), strides=(2, 2), padding='same', name = 'upsample_conv_{}'.format(i), activation = 'relu')(x)
        if use_batch_norm == True:
            x = BatchNormalization()(x)

    x = Conv2DTranspose(filters = 1, kernel_size = (12, 3), strides = (2, 2), padding='same', name = f'generator_Tconv_5', activation='tanh')(x)
    # x = LeakyReLU(alpha=0.2)(x)
    # x = keras.activations.tanh(x)
    
    generator_output = x 
    generator = Model([generator_input, label_input], generator_output, name = 'Generator')
    return generator

g_model = generator()
g_model.summary()
keras.utils.plot_model(g_model, "generator_plot.png", show_shapes=True)

def discriminator(n_classes = 5):
    
    discriminator_filters = [64, 128, 256, 512, 1024]

    label_input = Input(shape=(1,), dtype='int32', name='discriminator_label_input')
    label_em = Embedding(n_classes, n_classes * 10)(label_input)
    label_em = Dense(32768)(label_em)
    label_em = Reshape((256,128, 1))(label_em)

    discriminator_input = Input(shape=(256,128,1), name='discriminator_input')
    x = Concatenate()([discriminator_input, label_em])

    for i in range(5):
        x = Conv2D(filters = discriminator_filters[i], kernel_size = (12,3), strides = (2,2), padding = 'same', name = f'discriminator_conv_{i}')(x)
        x = LeakyReLU(alpha = 0.2)(x)

    x = Flatten()(x)
    #x = Dropout(0.2)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = Reshape((x,))(x)
    
    discriminator_output = Dense(1)(x)
    discriminator = Model([discriminator_input, label_input], discriminator_output, name = 'Discriminator')
    return discriminator

d_model = discriminator()
d_model.summary()
keras.utils.plot_model(d_model, "discriminator_plot.png", show_shapes=True)