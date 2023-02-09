# Module to download the dataset.
import ctifgan
import wgan_gp
import utils
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
audio_size_samples = 16384
foldername = 'gstep-2'
clipBelow = -10
mse_weight = 1e-4

def normalize(specs):
    max_value = tf.reduce_max(specs)
    normalized_spec = specs / max_value
    return normalized_spec, max_value


def train_model(n_batches = 100000,
                sampling_rate = 16000,
                batch_size = 32,
                audio_path = audio_dir,
                checkpoints_path = checkpoints_path,
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 5000,
                save_frequency = 10000,
                latent_dim = 100,
                discriminator_learning_rate = 1e-4,
                generator_learning_rate = 1e-4):

    n_classes = utils.get_n_classes(audio_path)
    #create the dataset from the class folders in '/audio'
    audio, labels = utils.create_dataset(audio_path, sampling_rate, checkpoints_path, audio_size_samples=audio_size_samples)
    wave = tf.reshape(tf.cast(audio,tf.float32),(-1,16384))
    specs = tfio.audio.spectrogram(wave, nfft=254, window=256, stride=64)
    normalized_spec, max_value = normalize(specs)
    log_spec = tf.math.log(tf.clip_by_value(t=normalized_spec, clip_value_min=tf.exp(-10.0), clip_value_max=float("inf")))
    log_spec = log_spec/(-clipBelow/2)+1
    specs = tf.reshape(log_spec, (-1, 256, 128, 1))
    specs = np.asarray(specs)

    #build the discriminator
    discriminator = ctifgan.discriminator(n_classes = n_classes)
    #build the generator
    generator = ctifgan.generator(latent_dim = latent_dim,
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
    checkpoints_path = utils.create_date_folder(checkpoints_path, name=foldername)
    
    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    utils.write_parameters(n_batches, batch_size, audio_path, checkpoints_path, path_to_weights, max_value,
                           resume_training, override_saved_model, synth_frequency, save_frequency, latent_dim, use_batch_norm, 
                           discriminator_learning_rate, generator_learning_rate, discriminator_extra_steps, mse_weight)


    #load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_weights}')
        gan.load_weights(path_to_weights)

    #train the gan for the desired number of batches
    gan.train(audio = audio, labels = labels, specs=specs, batch_size = batch_size, batches = n_batches, 
                 max_value = max_value, synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 sampling_rate = sampling_rate, n_classes = n_classes, mse_weight = mse_weight)


if __name__ == '__main__':
    audio_dir = '/home/rein/Downloads/allclass/footstep/'
    checkpoints_path = 'footstep_checkpoints/'

    train_model(n_batches = 100000,
                sampling_rate = 16000,
                batch_size = 32,
                audio_path = audio_dir,
                checkpoints_path = checkpoints_path,
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 5000,
                save_frequency = 10000,
                latent_dim = 100,
                use_batch_norm = True,
                discriminator_learning_rate = 1e-4,
                generator_learning_rate = 1e-4,
                discriminator_extra_steps = 1)
