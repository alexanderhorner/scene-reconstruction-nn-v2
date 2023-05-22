import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import math
import datetime
import time
from IPython import display
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', default='special-30k-4views')
parser.add_argument('--runname', default='BS8-special-chatGPT-4cameraviews-test2')
parser.add_argument('--batchsize', default='8')
parser.add_argument('--learningrate', default='2e-4')
args = parser.parse_args()

# DATASET
RUN_TIMEDATA = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# BUFFER_SIZE = 26000
BUFFER_SIZE = 4000
BATCH_SIZE = int(args.batchsize)
DATASET_NAME = args.dataset
RUN_NAME = args.runname
MODEL_NAME = 'cGAN2'
DATASET_PATH = 'data/' + DATASET_NAME + '/'
CHECKPOINT_DIR = 'training_checkpoints/' + MODEL_NAME + '/' + DATASET_NAME + '/' + RUN_NAME + '/'
LOG_DIR = 'logs/' + MODEL_NAME + '/' + DATASET_NAME + '/' + RUN_TIMEDATA + '-' + RUN_NAME + '/'
PRGRESS_RESULTS_DIR = 'progressResults/' + MODEL_NAME + '/' + DATASET_NAME + '/' + RUN_TIMEDATA + '-' + RUN_NAME + '/'

# Load dataset
print('Loading dataset...')
print("BATCH_SIZE: " + str(BATCH_SIZE))
print("MODEL_NAME: " + str(MODEL_NAME))
print("RUN_NAME: " + str(RUN_NAME))
print("DATASET_NAME: " + str(DATASET_NAME))
print("LEARNING_RATE: " + str(float(args.learningrate)))

# Image load function  
def loadImage(image_file_path):
  image = tf.io.read_file(image_file_path)
  image = tf.io.decode_png(image, channels=3)
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
plt.close()

# organize dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print('Loaded dataset.')

# MODEL ARCHITECTURE
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

# generator loss
LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

# OPTIMIZER
generator_optimizer = tf.keras.optimizers.Adam(float(args.learningrate), beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(float(args.learningrate), beta_1=0.5)

#checkpoints
checkpoint_prefix = CHECKPOINT_DIR + 'ckpt'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# TRAINING
@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

def fit(train_ds, test_ds):
  step = 0
  
  for epoch in range(100):
    start = time.time()
    # Train
    
    print('Epoch', str(epoch + 1), 'going on....')

    for input_image, target in tqdm(train_ds):
      step += 1
      train_step(input_image, target)

      if (step) % 125 == 0:
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
          predictedImg = generator(inputimg4d, training=False)
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
        plt.savefig(PRGRESS_RESULTS_DIR + 'step' + str(step) + '(epoche' + str(epoch + 1) + ')' + '.png')
        plt.close()


    print('Completed.')

    checkpoint.save(file_prefix=checkpoint_prefix)
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        
# Restore the latest checkpoint in checkpoint_dir
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
checkpoint.restore(latest_checkpoint)
print('Restored checkpoint')
print(latest_checkpoint)

fit(train_dataset, test_dataset)

print('END')