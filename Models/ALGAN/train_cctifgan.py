import ctifgan
import wgan_gp
import utils
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
audio_size_samples = 16384
foldername = 'dstep-5'

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
                use_batch_norm = False,
                discriminator_learning_rate = 0.00004,
                generator_learning_rate = 0.00004,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0):

    n_classes = utils.get_n_classes(audio_path)
    #create the dataset from the class folders in '/audio'
    audio, labels = utils.create_dataset(audio_path, sampling_rate, checkpoints_path, audio_size_samples=audio_size_samples)

    #build the discriminator
    discriminator = ctifgan.discriminator(n_classes = n_classes)
    #build the generator
    generator = ctifgan.generator(latent_dim = latent_dim,
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
    checkpoints_path = utils.create_date_folder(checkpoints_path, name=foldername)
    
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
    gan.train(x = audio, y = labels, batch_size = batch_size, batches = n_batches, 
                 synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 sampling_rate = sampling_rate, n_classes = n_classes)


if __name__ == '__main__':
    audio_dir = '/home/rein/OneDrive/selected_sounds/processed/Footstep-processed/16khz/'
    checkpoints_path = 'footstep_checkpoints/'

    train_model(n_batches = 100000,
                sampling_rate = 16000,
                batch_size = 16,
                audio_path = audio_dir,
                checkpoints_path = checkpoints_path,
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 2000,
                save_frequency = 2000,
                latent_dim = 100,
                use_batch_norm = True,
                discriminator_learning_rate = 1e-4,
                generator_learning_rate = 1e-4,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0)