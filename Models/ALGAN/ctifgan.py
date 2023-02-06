from tensorflow.keras.layers import Lambda, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization, Dropout, Cropping1D
from tensorflow.keras.models import Model
#from tensorflow.keras import backend as K
import keras.activations
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_io as tfio
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import utils
#import kapre

#Label embeding using the method in https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
#n_classes = 5
audio_dim =16384
n_fft = 254
hop_length= 64
win_length = 254
iterations = 50
clipBelow = -10
max_value = 69.37411499023438

def normalize(specs):
    max_value = tf.reduce_max(specs)
    normalized_spec = specs / max_value
    return normalized_spec, max_value

# checkpoints_path = 'allclass_checkpoints/'
# audio_path = '/home/rein/OneDrive/selected_sounds/processed/allclass/'
# audio, labels = utils.create_dataset(audio_path, 16000, checkpoints_path, audio_size_samples=16384)
# wave = tf.reshape(tf.cast(audio,tf.float32),(-1,16384))
# specs = tfio.audio.spectrogram(wave, nfft=254, window=254, stride=64)
# max_value = tf.reduce_max(tf.abs(specs))

class Spectrogram(Layer):
    def __init__(self, n_fft, win_length, hop_length, **kwargs) :
        super(Spectrogram, self).__init__(trainable=False)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
    def call(self, x):
        spectrogram = tfio.audio.spectrogram(x, nfft=self.n_fft, window=self.win_length, stride=self.hop_length)
        normalized_spec = spectrogram / max_value
        log_spec = tf.math.log(tf.clip_by_value(t=normalized_spec, clip_value_min=tf.exp(-10.0), clip_value_max=float("inf")))
        log_spec = log_spec/(-clipBelow/2)+1
        return log_spec
    def get_config(self):
        config = super(Spectrogram, self).get_config()
        config.update({'n_fft': self.n_fft,'win_length': self.win_length,'hop_length': self.hop_length,})
        return config

class inv_spec(Layer):
    def __init__(self, n_fft, win_length, hop_length, **kwargs) :
        super(inv_spec, self).__init__(trainable=False)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
    def call(self, x):
        de_log_spec = tf.math.exp(5*(x-1))
        de_norm_spec = de_log_spec * max_value
        x = tf.squeeze(de_norm_spec,-1)
        waveform = tfio.audio.inverse_spectrogram(x, nfft=self.n_fft, window=self.win_length, stride=self.hop_length, iterations=iterations)
        waveform = waveform / tf.reduce_max(tf.abs(waveform))
        #waveform = tf.pad(waveform, [[0,0], [0,63]])
        waveform = tf.slice(waveform, [0, 0], [-1, audio_dim])
        waveform = tf.expand_dims(waveform, -1)
        return waveform
    def get_config(self):
        config = super(inv_spec, self).get_config()
        config.update({'n_fft': self.n_fft,'win_length': self.win_length,'hop_length': self.hop_length})
        return config

def generator(latent_dim=100, use_batch_norm=True, n_classes=1):

    generator_filters = [512, 256, 128, 64, 1]

    label_input = Input(shape=(1,), dtype='int32', name='generator_label_input')
    label_em = Embedding(n_classes, 50, name = 'label_embedding')(label_input)
    label_em = Dense(32, name = 'label_dense')(label_em)
    label_em = Reshape((8, 4, 1), name = 'label_reshape')(label_em)
    
    generator_input = Input(shape=(latent_dim,), name='generator_input')
    x = generator_input
    x = Dense(16384, name='generator_input_dense')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((8, 4, 512), name='generator_input_reshape')(x)
    # Concatenate embedding
    x = Concatenate()([x, label_em])

    for i in range(4):
        x = Conv2DTranspose(filters=generator_filters[i], kernel_size=(12, 3), strides=(2, 2), padding='same', name = 'upsample_conv_{}'.format(i), activation = 'relu')(x)
        if use_batch_norm == True:
            x = BatchNormalization()(x)

    x = Conv2DTranspose(filters = 1, kernel_size = (12, 3), strides = (2, 2), padding='same', name = f'generator_Tconv_5', activation='tanh')(x)
    # # Inverse STFT
    #tf.keras.utils.save_img('image.png', x[0], data_format=None, file_format=None, scale=True)
    # x = Reshape((256, 128))(x)
    x = inv_spec(n_fft=n_fft, win_length=win_length, hop_length=hop_length, trainable=False)(x)
    #x = tf.keras.layers.ZeroPadding1D((0,63), trainable = False)(x)
    #x = Lambda(lambda x: tf.slice(x, [0, 0, 0], [-1, audio_dim, -1]))(x)
    # x = Reshape((16384, 1))(x)
    #x = Cropping1D(cropping=(0,190), trainable = False)(x)
    #x = Lambda(lambda x: x[:,:16384,:])(x)
    #x = Reshape((16384, 1))(x)

    generator_output = x
    generator = Model([generator_input, label_input], generator_output, name = 'Generator')
    return generator

g_model = generator()
#g_model.summary()
keras.utils.plot_model(g_model, "generator_plot.png", show_shapes=True)


def discriminator(n_classes=5):
    
    discriminator_filters = [64, 128, 256, 512, 1024]

    label_input = Input(shape=(1,), dtype='int32', name='discriminator_label_input')
    label_em = Embedding(n_classes, 50)(label_input)
    label_em = Dense(32768)(label_em)
    label_em = Reshape((256,128,1))(label_em)

    discriminator_input = Input(shape=(audio_dim,1), name='discriminator_input')
    x = Reshape((1, audio_dim))(discriminator_input)
    x = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, trainable=False)(x)
    x = Reshape((256,128,1))(x)
    # x = kapre.STFT(n_fft=n_fft, win_length=None, hop_length=hop_length, window_name='hann_window', pad_end=True, input_shape=discriminator_input.shape, trainable=True)(discriminator_input)
    # x = kapre.Magnitude()(x)
    x = Concatenate()([x, label_em])

    for i in range(5):
        x = Conv2D(filters = discriminator_filters[i], kernel_size = (12,3), strides = (2,2), padding = 'same', name = f'discriminator_conv_{i}')(x)
        #x = tfa.layers.SpectralNormalization(Conv2D(filters = discriminator_filters[i], kernel_size = (12,3), strides = (2,2), padding = 'same', name = f'discriminator_conv_{i}'))(x)
        x = LeakyReLU(alpha = 0.2)(x)
    x = Flatten()(x)
    
    discriminator_output = Dense(1)(x)
    discriminator = Model([discriminator_input, label_input], discriminator_output, name = 'Discriminator')
    return discriminator

d_model = discriminator()
#d_model.summary()
keras.utils.plot_model(d_model, "discriminator_plot.png", show_shapes=True)