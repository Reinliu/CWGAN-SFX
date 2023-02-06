import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import librosa
import soundfile as sf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from ctifgan import *
#Baseline WGANGP model directly from the Keras documentation: https://keras.io/examples/generative/wgan_gp/
clip_threshold = 1.0
import pandas as pd
losses = []

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
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, labels], training=True)
        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_batch(self, x, y, batch_size):
        #get a random indexes for the batch
        idx = np.random.randint(0, x.shape[0], batch_size)
        real_images = x[idx]
        labels = y[idx]
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.

        # BCE loss:
        # bce = tf.keras.losses.BinaryCrossentropy()
        # for i in range(self.d_steps):
        # # Train the generator now.
        # # Get the latent vector
        #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        #     with tf.GradientTape() as tape:
        #         # Generate fake images using the generator
        #         generated_images = self.generator([random_latent_vectors, labels], training=True)
        #         # Get the discriminator logits for fake images
        #         gen_img_logits = self.discriminator([generated_images, labels], training=True)
        #         # Calculate the generator loss
        #         g_loss = bce(tf.ones_like(gen_img_logits), gen_img_logits)

        #     # # Get the gradients w.r.t the generator loss
        #     gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        #     # # Update the weights of the generator using the generator optimizer
        #     self.g_optimizer.apply_gradients(
        #         zip(gen_gradient, self.generator.trainable_variables)
        #     )
        
        # # Train the discriminator now
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # with tf.GradientTape() as tape:
        #     fake_images = self.generator([random_latent_vectors, labels], training=True)
        #     fake_logits = self.discriminator([fake_images, labels], training=True)
        #     real_logits = self.discriminator([real_images, labels], training=True)
        #     d_loss_real = bce(tf.ones_like(real_logits), real_logits)
        #     d_loss_fake = bce(tf.zeros_like(fake_logits), fake_logits)
        #     d_loss = d_loss_real + d_loss_fake
    
        # # Get the gradients w.r.t the discriminator loss
        # d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # # Update the weights of the discriminator using the discriminator optimizer
        # self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # WGAN loss:
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, labels], training=True)
                # fake_images = tf.reshape(fake_images, (batch_size, 1, 256, 128))
                # fake_images = tfio.audio.inverse_spectrogram(fake_images, nfft=n_fft, window=win_length, stride=hop_length)
                # fake_images = tf.slice(fake_images, [0, 0, 0], [-1, -1, audio_dim])
                # fake_images = tf.reshape(fake_images, (batch_size, 16384, 1))
                # Get the logits for the fake images
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
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            
        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, labels], training=True)
            # generated_images = tf.reshape(generated_images, (batch_size, 1, 256, 128))
            # generated_images = tfio.audio.inverse_spectrogram(generated_images, nfft=n_fft, window=win_length, stride=hop_length)
            # generated_images = tf.slice(generated_images, [0, 0, 0], [-1, -1, audio_dim])
            # generated_images = tf.reshape(generated_images, (batch_size, 16384, 1))
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #sf.write('saved_wave.wav', tf.squeeze(generated_images[0]).numpy(), samplerate=16000)

        #print('checking discriminator gradients', d_gradient)
        #print('checking generator gradients', gen_gradient)
        # Create a new GradientTape
        # with tf.GradientTape() as tape:
        #     # Generate fake images using the generator
        #     generated_images = self.generator([random_latent_vectors, labels], training=True)
        #     # Get the discriminator logits for fake images
        #     gen_img_logits = self.discriminator([generated_images, labels], training=True)
        #     # Calculate the generator loss
        #     g_loss = self.g_loss_fn(gen_img_logits)

        #     # Compute the gradients for the last layer of the generator
        # layer_name = "upsample_conv_0"
        # gradients = tape.gradient(g_loss, self.generator.get_layer(layer_name).trainable_variables)

        return d_loss, g_loss
    
    def train(self, x, y, batch_size, batches, synth_frequency, save_frequency,
              sampling_rate, n_classes, checkpoints_path, override_saved_model):

        for batch in range(batches):
            start_time = time.time()
            d_loss, g_loss = self.train_batch(x, y, batch_size)
            losses.append([d_loss, g_loss])
            end_time = time.time()
            time_batch = (end_time - start_time)
            print(f'Batch: {batch} == Batch size: {batch_size} == Time elapsed: {time_batch:.2f} == d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
            
            #This works as a callback
            if batch % synth_frequency == 0 :
                print(f'Synthesising audio at batch {batch}. Path: {checkpoints_path}/synth_audio')
                random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
                for i in range (n_classes):
                    generated_audio = self.generator([random_latent_vectors, np.array(i).reshape(-1,1)])
                    sf.write(f'{checkpoints_path}/synth_audio/{batch}_batch_synth_class_{i}.wav', 
                                             tf.squeeze(generated_audio).numpy(), samplerate = sampling_rate)
                    gen_spec = tfio.audio.spectrogram(tf.squeeze(generated_audio,axis=-1), nfft=254, window=254, stride=64)
                    normalized_spec = gen_spec / 69.37411499023438
                    log_spec = tf.math.log(tf.clip_by_value(t=normalized_spec, clip_value_min=tf.exp(-10.0), clip_value_max=float("inf")))
                    log_spec = log_spec/(-clipBelow/2)+1
                    plt.imsave(f'{checkpoints_path}/synth_audio/{batch}_batch_synth_class_{i}.png', tf.squeeze(log_spec).numpy())
                    #print(generated_audio.shape, tf.reduce_mean(generated_audio), gen_spec.shape, tf.reduce_mean(gen_spec))
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