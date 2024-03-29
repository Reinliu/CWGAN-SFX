# Module to download the dataset.
import ctifgan
import wgan_gp
import utils
from tensorflow.keras.optimizers import Adam
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
audio_dim = 16384*4
foldername = 'gstep-1'
clipBelow = -10
mse_weight = 1e-2

def normalize(specs):
    max_value = tf.reduce_max(specs)
    normalized_spec = specs / max_value
    return normalized_spec, max_value

def extract_specs(audio, audio_dim):
    wave = tf.reshape(tf.cast(audio,tf.float32),(-1,audio_dim))
    if audio_dim == 16384*4:
        specs = tfio.audio.spectrogram(wave, nfft=510, window=256, stride=128)
    elif audio_dim == 16384:
        specs = tfio.audio.spectrogram(wave, nfft=254, window=256, stride=64)
    normalized_spec, max_value = normalize(specs)
    log_spec = tf.math.log(tf.clip_by_value(t=normalized_spec, clip_value_min=tf.exp(-10.0), clip_value_max=float("inf")))
    log_spec = log_spec/(-clipBelow/2)+1
    if audio_dim == 16384*4:
        specs = tf.reshape(log_spec, (-1, 512, 256, 1))
    elif audio_dim == 16384:
        specs = tf.reshape(log_spec, (-1, 256, 128, 1))
    specs = np.asarray(specs)

    return specs, max_value

def train_model(sampling_rate = 16000,
                n_batches = 10000,
                batch_size = 128,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints/',
                resume_training = False,
                path_to_weights = 'checkpoints/model_weights.h5',
                override_saved_model = False,
                synth_frequency = 200,
                save_frequency = 200,
                latent_dim = 100,
                discriminator_learning_rate = 0.00004,
                generator_learning_rate = 0.00004,
                discriminator_extra_steps = 5):

    n_classes = utils.get_n_classes(audio_path)
    #create the dataset from the class folders in '/audio'
    audio, labels = utils.create_dataset(audio_path, sampling_rate, checkpoints_path, audio_size_samples=audio_dim)
    specs, max_value = extract_specs(audio, audio_dim)

    #build the discriminator
    discriminator = ctifgan.discriminator(audio_dim = audio_dim, n_classes = n_classes)
    #build the generator
    generator = ctifgan.generator(latent_dim = latent_dim, audio_dim = audio_dim, n_classes = n_classes)
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
    checkpoints_path = utils.create_date_folder(checkpoints_path, name=foldername)
    
    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    utils.write_parameters(n_batches, batch_size, audio_path, checkpoints_path, path_to_weights, max_value,
                           resume_training, override_saved_model, synth_frequency, save_frequency, latent_dim, 
                           discriminator_learning_rate, generator_learning_rate, discriminator_extra_steps, mse_weight)


    #load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_weights}')
        gan.load_weights(path_to_weights)

    #train the gan for the desired number of batches
    gan.train(audio = audio, labels = labels, specs=specs, audio_dim = audio_dim, batch_size = batch_size, batches = n_batches, 
                 max_value = max_value, synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 sampling_rate = sampling_rate, n_classes = n_classes, mse_weight = mse_weight)


if __name__ == '__main__':
    audio_dir = '/home/rein/Downloads/development-dataset/raining/'
    checkpoints_path = 'challenge_checkpoints/'

    train_model(n_batches = 100000,
                sampling_rate = 16000,
                batch_size = 4,
                audio_path = audio_dir,
                checkpoints_path = checkpoints_path,
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 2000,
                save_frequency = 2000,
                latent_dim = 100,
                discriminator_learning_rate = 1e-4,
                generator_learning_rate = 1e-4,
                discriminator_extra_steps = 1)
