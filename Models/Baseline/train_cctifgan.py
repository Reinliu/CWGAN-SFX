# Module to download the dataset.
import librosa
import numpy as np
from ourLTFATStft import LTFATStft
import ltfatpy
from modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad # This function is not implemented for one sided stfts with the phase method on ltfatpy
from pghipy import get_default_window, calculate_synthesis_window
from pghipy import stft, pghi, istft
import ctifgan
import wgan_gp
import utils
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

sample_rate = 16000
audio_dim = 16384
clipBelow = -10

def normalize(db_spec):
    normalized_spec = (db_spec - tf.reduce_min(db_spec)) / (tf.reduce_max(db_spec) - tf.reduce_min(db_spec))
    return normalized_spec

def denormalize(normalized_spec):
    denormalized_spec = normalized_spec * 69.374115
    return denormalized_spec

def generate_spectrograms_and_derivs_from(audio_signals):
    fft_hop_size = 128
    fft_window_length = 512
    L = 16384
    anStftWrapper = LTFATStft()
    spectrograms = np.zeros([len(audio_signals), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    # gs = {'name': 'gauss', 'M': 512}
    
    for index, audio_signal in enumerate(audio_signals):
        realDGT = anStftWrapper.oneSidedStft(signal=audio_signal, windowLength=fft_window_length, hopSize=fft_hop_size)
        spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e**clipBelow, normalize=True)
        spectrograms[index] = spectrogram  
    return spectrograms

def train_model(n_batches = 10000,
                batch_size = 128,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints/',
                resume_training = False,
                path_to_weights = 'checkpoints/model_weights.h5',
                override_saved_model = False,
                synth_frequency = 200,
                save_frequency = 200,
                latent_dim = 100,
                use_batch_norm = False,
                discriminator_learning_rate = 0.00004,
                generator_learning_rate = 0.00004,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0):

    n_classes = utils.get_n_classes(audio_dir)
    audio, labels = utils.create_dataset(audio_dir, sample_rate, checkpoints_path, audio_size_samples=audio_dim)

    wave = tf.reshape(tf.cast(audio,tf.float32),(-1,16384))
    specs = tfio.audio.spectrogram(wave, nfft=254, window=254, stride=64)
    log_spec = tf.math.log(tf.clip_by_value(t=normalize(specs), clip_value_min=tf.exp(-10.0), clip_value_max=float("inf")))
    log_spec = log_spec/(-clipBelow/2)+1
    print(tf.reduce_min(log_spec), tf.reduce_max(log_spec))
    specs = tf.reshape(log_spec, (-1, 256, 128, 1))
    specs = np.asarray(specs)

    #build the discriminator
    discriminator = ctifgan.discriminator(n_classes = n_classes)
    #build the generator
    generator = ctifgan.generator(z_dim = latent_dim,
                                                use_batch_norm = use_batch_norm,
                                                n_classes = n_classes)
    #set the optimizers
    discriminator_optimizer = Adam(learning_rate = discriminator_learning_rate, beta_1=0.5, beta_2=0.9)
    generator_optimizer = Adam(learning_rate = generator_learning_rate, beta_1=0.5, beta_2=0.9)
    
    #build the gan
    gan = wgan_gp.WGANGP(latent_dim=latent_dim, discriminator=discriminator, generator=generator,
                    n_classes = n_classes, discriminator_extra_steps = discriminator_extra_steps,
                    d_optimizer = discriminator_optimizer, g_optimizer = generator_optimizer)

    # Compile the wgan model
    gan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer)

    #make a folder with the current date to store the current session to
    #avoid overriding past synth audio files and checkpoints
    checkpoints_path = utils.create_date_folder(checkpoints_path)
    
    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    utils.write_parameters(n_batches, batch_size, audio_path, checkpoints_path, path_to_weights, 
                           resume_training, override_saved_model, synth_frequency, save_frequency, latent_dim, use_batch_norm, 
                           discriminator_learning_rate, generator_learning_rate, discriminator_extra_steps,phaseshuffle_samples)

    #load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_weights}')
        gan.load_weights(path_to_weights)

    #train the gan for the desired number of batches
    gan.train(x = specs, y = labels, batch_size = batch_size, batches = n_batches, 
                 synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 n_classes = n_classes)

if __name__ == '__main__':
    audio_dir = '/home/rein/Downloads/allclass/footstep/'
    checkpoints_path = 'footstep_checkpoints/'

    # spectrograms = generate_spectrograms_and_derivs_from(audio)
    # spectrograms = spectrograms/(-clipBelow/2)+1
    # specs = spectrograms
    # specs = np.expand_dims(specs, axis=-1)
    # specs = specs[:, :256, :]
    # print(specs.shape, specs.dtype)

    train_model(n_batches = 100000,
                batch_size = 32,
                audio_path = audio_dir,
                checkpoints_path = checkpoints_path,
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 500,
                save_frequency = 2000,
                latent_dim = 100,
                use_batch_norm = True,
                discriminator_learning_rate = 1e-4,
                generator_learning_rate = 1e-4,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0)