import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import soundfile as sf
import tensorflow_io as tfio
import numpy as np

checkpoints_path = 'checkpoints'
path_to_generator = 'generator_path/generator.h5'
path_to_labels = 'labels_path/label_names.json'
path_to_save = ''
max_value = 114.14022827148438 # The max value extracted from spectrograms. Obtained from training_parameters.txt

z_dim = 100
sample_rate = 16000

#load the generator
generator = tf.keras.models.load_model(path_to_generator)
#read the labels from the generated dictionary during training
with open(path_to_labels) as json_file:
    label_names = json.load(json_file)
label_names

#create noise and label
n_samples_label = 1000

for synth_type in tqdm(label_names):
    noise  = tf.random.normal(shape=(n_samples_label, z_dim))
    label_synth = tf.constant(int(synth_type), shape=(n_samples_label,1))
    synth_audio = generator.predict([noise, label_synth])
    print(synth_audio.shape)
    de_log_spec = tf.math.exp(5*(synth_audio-1))
    generated_audio = de_log_spec * max_value
    generated_audio = tf.reshape(generated_audio, (-1, 256, 128))
    generated_audio = tfio.audio.inverse_spectrogram(generated_audio, nfft=254, stride=64, window=254, iterations=50)
    generated_audio = generated_audio / tf.reduce_max(tf.abs(generated_audio))
    generated_audio = generated_audio[:,:sample_rate]
    generated_audio = np.asarray(generated_audio, dtype = 'float32')
    print(generated_audio.dtype)
    print(generated_audio.shape, generated_audio.max(), generated_audio.min())
    for i in range(n_samples_label):
        sf.write(f'{path_to_save}/{label_names[synth_type]}_{i}.wav', data = generated_audio[i]/generated_audio[i].max(), samplerate = sample_rate) 