import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import soundfile as sf
import tensorflow_io as tfio
import numpy as np
from tensorflow.keras.layers import Layer

path_to_generator = '/home/rein/Documents/layered_cctifgan/footstep_checkpoints/05-02-2023_11_dstep-5/generator.h5'
path_to_labels = '/home/rein/Documents/layered_cctifgan/footstep_checkpoints/label_names.json'
path_to_save = '/home/rein/Documents/layered_cctifgan/generated_sounds/footstep/'

max_value = 69.37411499023438 # The max value extracted from spectrograms. Obtained from training_parameters.txt
audio_dim =16384
n_fft = 254
hop_length= 64
win_length = 256
iterations = 50
clipBelow = -10

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


z_dim = 100
sample_rate = 16000

# Load the generator. Including the custom layer
generator = tf.keras.models.load_model(path_to_generator, custom_objects={'inv_spec':inv_spec})

# read the labels from the generated dictionary during training
with open(path_to_labels) as json_file:
    label_names = json.load(json_file)
label_names

# create noise and label
n_samples_label = 1000

for synth_type in tqdm(label_names):
    noise = tf.random.normal(shape=(n_samples_label, z_dim))
    label_synth = tf.constant(int(synth_type), shape=(n_samples_label,1))
    synth_audio = generator.predict([noise, label_synth])
    for i in range(n_samples_label):
        sf.write(f'{path_to_save}/{label_names[synth_type]}_{i}.wav', data = np.squeeze(synth_audio[i]), samplerate = sample_rate) 