import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import math
import datetime
import time
from IPython import display
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', default='simplecars-norotation-fixedpos-whitebg-squarelines', )

args = parser.parse_args()








# DATASET

RUN_TIMEDATA = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


# BUFFER_SIZE = 60000
BUFFER_SIZE = 4000
# BATCH_SIZE = 256
BATCH_SIZE = 1


DATASET_NAME = args.dataset
RUN_NAME = "first-try"
MODEL_NAME = 'cGAN'
DATASET_PATH = 'src/data/' + DATASET_NAME + '/'
CHECKPOINT_DIR = 'training_checkpoints/' + MODEL_NAME + '/' + DATASET_NAME + '/'
LOG_DIR = 'logs/' + MODEL_NAME + '/' + DATASET_NAME + '/' + RUN_TIMEDATA + '-' + RUN_NAME + '/'
PRGRESS_RESULTS_DIR = 'progressResults/' + MODEL_NAME + '/' + DATASET_NAME + '/' + RUN_TIMEDATA + '-' + RUN_NAME + '/'



# Load dataset
print('Loading dataset...')

# Image load function  
def loadImage(image_file_path):
  image = tf.io.read_file(image_file_path)
  image = tf.io.decode_png(image, channels=3)
  # image = tf.image.resize(image, (64, 64))

  # # Split each image tensor into two tensors:
  # # - one with a real building facade image
  # # - one with an architecture label image 
  # w = tf.shape(image)[1]
  # w = w // 2
  # input_image = image[:, w:, :]
  # real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  image = tf.cast(image, tf.float32)

  image = normalizeImage(image)

  return image

def normalizeImage(image):
  image = (image / 127.5) - 1
  return image

def reverseNormalizeImageAndConvertToUINT8(image):
  image = (image +1) * 127.5
  image = tf.cast(image, tf.uint8)
  return image

def getImageArray(pathToImageSetFolder):
  inputImage = loadImage(pathToImageSetFolder + '/camera1.png')
  targetImage = loadImage(pathToImageSetFolder + '/cameraTarget.png')

  return inputImage, targetImage

# Get list of files
train_dataset = tf.data.Dataset.list_files(DATASET_PATH + '/train/*/')
test_dataset = tf.data.Dataset.list_files(DATASET_PATH + '/test/*/')
  
# call image load function for every file in list
train_dataset = train_dataset.map(getImageArray, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(getImageArray, num_parallel_calls=tf.data.AUTOTUNE)


# Display a few samples from the training set
f, axs = plt.subplots(3, 6, figsize=(12, 6))
for row in axs:
  for ax in row:
    ax.axis('off')
for i, (inputImage, targetImage) in enumerate(train_dataset.take(9)):
  placeInRow = i%3
  row = math.floor(i / 3)
  axs[row, placeInRow*2].imshow(reverseNormalizeImageAndConvertToUINT8(inputImage))
  axs[row, placeInRow*2+1].imshow(reverseNormalizeImageAndConvertToUINT8(targetImage))
  axs[row, placeInRow*2].title.set_text('Input  ---->')
  axs[row, placeInRow*2+1].title.set_text('Target')
tf.io.gfile.makedirs(PRGRESS_RESULTS_DIR)
plt.savefig(PRGRESS_RESULTS_DIR + '/trainingset-samples.png')

# organize dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print('Loaded dataset.')









# MODEL ARCHITECTURE
def make_generator_model():
  model = tf.keras.Sequential()
  model.add(layers.Dense(16*16*1024, use_bias = False, input_shape = (100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  print(model.output_shape)

  model.add(layers.Reshape((16, 16, 1024)))
  assert model.output_shape == (None, 16, 16, 1024)  # Note: None is the batch size


  model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 32, 32, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 64, 64, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 256, 256, 3)

  return model      

generator = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)




def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=[256, 256, 3]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

discriminator = make_discriminator_model()

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)









# set logging and checkpoints for training
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
summary_writer = tf.summary.create_file_writer(LOG_DIR)

#checkpoints
checkpoint_prefix = CHECKPOINT_DIR + 'ckpt'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)




# TRAINING

@tf.function
def train_step(images, step):
  noise = tf.random.normal([BATCH_SIZE, 100])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_loss', gen_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)



def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      print(f'Step: {step}')

    train_step(input_image, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 1000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      
    if (step + 1) % 100 == 0:
      # Display a few samples from the validation set
      f, axs = plt.subplots(3, 6, figsize=(12, 6))
      for row in axs:
        for ax in row:
          ax.axis('off')
      for i in range(6):
        picNumber = 5001 + i

        # make a prediction
        inputimg = loadImage(DATASET_PATH + '/val/' + str(picNumber) + '/camera1.png')
        inputimg4d = tf.expand_dims(inputimg, 0)
        # predictedImg = generator(inputimg4d, training=False)
        noise = tf.random.normal([BATCH_SIZE, 100])
        # noise = tf.random.Generator.from_seed(hash(inputimg4d.ref()))
        # noise = noise.normal(shape=[BATCH_SIZE, 100])
        predictedImg = generator(noise, training=False)
        predictedImg = tf.squeeze(predictedImg)
        targetImg = loadImage(DATASET_PATH + '/val/' + str(picNumber) + '/cameraTarget.png')


        placeInRow = i%2
        row = math.floor(i / 2)
        axs[row, placeInRow*3].imshow(reverseNormalizeImageAndConvertToUINT8(inputimg))
        axs[row, placeInRow*3+1].imshow(reverseNormalizeImageAndConvertToUINT8(targetImg))
        axs[row, placeInRow*3+2].imshow(reverseNormalizeImageAndConvertToUINT8(predictedImg))
        axs[row, placeInRow*3].title.set_text('Input  ---->')
        axs[row, placeInRow*3+1].title.set_text('Target  ---->')
        axs[row, placeInRow*3+2].title.set_text('Predicted')
      plt.savefig(PRGRESS_RESULTS_DIR + 'step' + str(int(step) + 1) + '.png')
        


latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
checkpoint.restore(latest_checkpoint)
print('Restored checkpoint')
print(latest_checkpoint)

fit(train_dataset, test_dataset, steps=40000000)



print('passed')