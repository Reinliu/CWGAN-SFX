import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import soundfile as sf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from ctifgan import *
import pandas as pd
losses = []

def mean_square_error(y_true, y_pred):
    diff = y_true - y_pred
    square_diff = tf.square(diff)
    sum = tf.reduce_sum(square_diff)
    mse = sum / y_true.shape[0]
    return mse

class WGANGP(keras.Model):
    def __init__(
        self,
        latent_dim,
        discriminator,
        generator,
        n_classes,
        discriminator_extra_steps=5,
        gp_weight=10,
        d_optimizer=tf.keras.optimizers.SGD(learning_rate=0.0004),
        g_optimizer=tf.keras.optimizers.SGD(learning_rate=0.0004)
    ):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = discriminator
        self.generator = generator
        self.n_classes = n_classes
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_optimizer=d_optimizer
        self.g_optimizer=g_optimizer

    def compile(self, d_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = self.discriminator_loss
        self.g_loss_fn = self.generator_loss      
    
    # Define the loss functions to be used for discriminator
    # This should be (fake_loss - real_loss)
    # We will add the gradient penalty later to this loss function
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss
    
    # Define the loss functions to be used for generator
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)
    
    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, labels], training=True)
        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_batch(self, audio, labels, specs, audio_dim, batch_size, max_value, mse_weight):
        #get a random indexes for the batch
        idx = np.random.randint(0, audio.shape[0], batch_size)
        real_images = specs[idx]
        labels = labels[idx]
        real_audios = audio[idx]
        if audio_dim == 16384*4:
            n_fft = 510
            hop_length = 128
            win_length = 256
        elif audio_dim == 16384:
            n_fft = 254
            hop_length = 64
            win_length = 254

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
            # Generate fake images using the generator
                generated_images = self.generator([random_latent_vectors, labels], training=True)
                de_log_spec = tf.math.exp(5*(generated_images-1))
                de_norm_spec = de_log_spec * max_value
                if audio_dim ==16384:
                    generated_audio = tf.reshape(de_norm_spec, (batch_size, 256, 128))
                elif audio_dim ==16384*4:
                    generated_audio = tf.reshape(de_norm_spec, (batch_size, 512, 256))
                generated_audio = tfio.audio.inverse_spectrogram(generated_audio, nfft=n_fft, stride=hop_length, window=win_length, iterations=50)
                generated_audio = generated_audio / (tf.reduce_max(tf.abs(generated_audio)) + 1e-50)
                generated_audio = generated_audio[:,:audio_dim]
                generated_audio = tf.reshape(generated_audio, (batch_size, audio_dim, 1))
                mse = mean_square_error(real_audios, generated_audio)
                # Get the discriminator logits for fake images
                gen_img_logits = self.discriminator([generated_images, labels], training=True)
                # Calculate the generator loss
                g_loss = self.g_loss_fn(gen_img_logits) + mse * mse_weight

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator([random_latent_vectors, labels], training=True)
            fake_logits = self.discriminator([fake_images, labels], training=True)
            # Get the logits for real images
            real_logits = self.discriminator([real_images, labels], training=True)
            # Calculate discriminator loss using fake and real logits
            d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        return d_loss, g_loss
    
    def train(self, audio, labels, specs, audio_dim, batch_size, max_value, batches, synth_frequency, save_frequency,
              sampling_rate, n_classes, checkpoints_path, override_saved_model, mse_weight):

        for batch in range(batches):
            if audio_dim == 16384*4:
                n_fft = 510
                hop_length = 128
                win_length = 256
            elif audio_dim == 16384:
                n_fft = 254
                hop_length = 64
                win_length = 254
            start_time = time.time()
            d_loss, g_loss = self.train_batch(audio, labels, specs, audio_dim, batch_size, max_value, mse_weight)
            losses.append([d_loss, g_loss])
            end_time = time.time()
            time_batch = (end_time - start_time)
            print(f'Batch: {batch} == Batch size: {batch_size} == Time elapsed: {time_batch:.2f} == d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
            
            #This works as a callback
            if batch % synth_frequency == 0 :
                print(f'Synthesising audio at batch {batch}. Path: {checkpoints_path}/synth_audio')
                random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
                for i in range (n_classes):
                    generated_signal = self.generator([random_latent_vectors, np.array(i).reshape(-1,1)])
                    plt.imsave(f'{checkpoints_path}/synth_audio/{batch}_batch_synth_class_{i}.png', tf.squeeze(generated_signal))
                    de_log_spec = tf.math.exp(5*(generated_signal-1))
                    de_norm_spec = de_log_spec * max_value
                    if audio_dim == 16384:
                        generated_audio = tf.reshape(de_norm_spec, (-1, 256, 128))
                    elif audio_dim == 16384*4:
                        generated_audio = tf.reshape(de_norm_spec, (-1, 512, 256))
                    generated_audio = tfio.audio.inverse_spectrogram(generated_audio, nfft=n_fft, stride=hop_length, window=win_length, iterations=50)
                    generated_audio = generated_audio / tf.reduce_max(tf.abs(generated_audio))
                    generated_audio = generated_audio[:,:audio_dim]
                    print(tf.reduce_max(generated_audio))
                    sf.write(f'{checkpoints_path}/synth_audio/{batch}_batch_synth_class_{i}.wav', 
                                             tf.squeeze(generated_audio).numpy(), samplerate = sampling_rate)
                print(f'Done.')
                
            if batch % save_frequency == 0:
                print(f'Saving the model at batch {batch}. Path: {checkpoints_path}')
                if override_saved_model == False:
                    self.generator.save(f'{checkpoints_path}/{batch}_batch_generator.h5')
                    self.discriminator.save(f'{checkpoints_path}/{batch}_batch_discriminator.h5')
                    self.save_weights(f'{checkpoints_path}/{batch}_batch_weights.h5')
                    loss_df = pd.DataFrame(losses, columns=['d_loss', 'g_loss'])
                    loss_df.to_csv(f'{checkpoints_path}/losses.csv', index=False)
                else:
                    self.generator.save(f'{checkpoints_path}/generator.h5')
                    self.discriminator.save(f'{checkpoints_path}/discriminator.h5')
                    self.save_weights(f'{checkpoints_path}/model_weights.h5')
                    loss_df = pd.DataFrame(losses, columns=['d_loss', 'g_loss'])
                    loss_df.to_csv(f'{checkpoints_path}/losses.csv', index=False)
                print(f'Model saved.')
