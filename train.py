#----------IMPORT PYTHON PACKAGES----------#
import cityscapesscripts.helpers.labels as labels
import numpy as np
import matplotlib.image as Image
from datetime import datetime
import argparse
import sys
import os

#----------IMPORT MODELS AND UTILITIES----------#
import unet, bayes_segnet, deeplabv3plus, fast_scnn
import datasets

#----------CONSTANTS----------#
CITYSCAPES_LABELS = [label for label in labels.labels if -1 < label.trainId < 255]
CITYSCAPES_COLORS = [label.color for label in CITYSCAPES_LABELS]
CITYSCAPES_IDS = [label.trainId for label in CITYSCAPES_LABELS]
CLASSES=19

#----------ARGUMENTS----------#
parser = argparse.ArgumentParser(prog='train.py', description="Start training a semantic segmentation model")
parser.add_argument("-m", "--model", help="Specify the model you wish to use: OPTIONS: unet, bayes_segnet, deeplabv3+, fastscnn")
parser.add_argument("-r", "--resume", help="Resume the training, specify the weights file path of the format weights-[epoch]-[val-acc]-[val-loss]-[datetime].hdf5")
parser.add_argument('-p',"--path", help="Specify the root folder for cityscapes dataset, if not used looks for CITYSCAPES_DATASET environment variable")
parser.add_argument('-c',"--coarse", help="Use the coarse images", action="store_true")
parser.add_argument('-t',"--target-size", help="Set the image size for training, should be a elements of a tuple x,y,c")
parser.add_argument('-b',"--batch", help="Set the batch size", default=8, type=int)
parser.add_argument('-e',"--epochs", help="Set the number of Epochs", default=1000, type=int)

models = ['unet', 'bayes_segnet', 'deeplabv3+', 'fastscnn']

args = parser.parse_args()
#Check models
if args.model not in models:
    sys.exit("No Model was Chosen or an invalid Model was chosen")
else:
    model_name = args.model
#Check the CITYSCAPES_ROOT path
if os.path.isdir(args.path):
    CITYSCAPES_ROOT = args.path
elif 'CITYSCAPES_DATASET' in os.environ:
    CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_DATASET')
else:
    sys.exit("No valid path for Cityscapes Dataset given")
#Now do the target size
if args.target_size is None:
    if model_name == 'fastscnn':
        target_size = (512, 1024, 3)
    else:
        target_size = (1024, 1024, 3)
else:
    target_size = tuple(int(i) for i in args.target_size.split(','))
#Get the Batch Size
batch_size = args.batch
#Get the number of epochs
epochs = args.epochs
#Check if the resume has been used, if so check the path
weights_path = args.resume
initial_epoch = 0
if args.resume is not None and os.path.isfile(args.resume):
    filename_split = os.path.basename(args.resume).split('-')
    initial_epoch = filename_split[1]
    time = filename_split[-1].replace('.hdf5', '')

#----------IMPORT TENSORFLOW AND CONFIGURE IT----------#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

#Set Memory Growth to alleviate memory issues
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
#Change policy to fp16 to use Tensor Cores
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#Set the strategy which will enable Multi-GPU
strategy = tf.distribute.MirroredStrategy()

#----------CREATE DATASETS----------#
train_ds = datasets.create_dataset(CITYSCAPES_ROOT, batch_size, epochs, target_size)
val_ds = datasets.create_dataset(CITYSCAPES_ROOT, batch_size, epochs, target_size, train=False)

#----------CREATE MODEL AND BEGIN TRAINING
time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
with strategy.scope():
    #Depending on model chosen, get the right model and create the optimizer for it
    if model_name == 'unet':
        model = unet.model(input_size=target_size)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    elif model_name == 'bayes_segnet':
        model = bayes_segnet.model(num_classes=CLASSES, input_size=target_size)
        optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=0.001)
    elif model_name == 'fastscnn':
        model = fast_scnn.model(input_size=target_size)
        optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=keras.optimizers.schedules.PolynomialDecay(0.045, epochs, power=0.9))
    elif model_name == 'deeplabv3+':  
        model = deeplabv3plus.model(input_size=target_size, num_classes=CLASSES)
        optimizer = keras.optimizers.SGD(learning_rate=tf.optimizers.schedules.PolynomialDecay(0.007, epochs, power=0.9))

    #Whatever the model is, compile it
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=CLASSES, name="meanIoU")]
                )

    #Ensure directories are made for callbacks
    if not os.path.isdir('./logs/'+model_name+'/checkpoints'):
        os.makedirs('./logs/'+model_name+'/checkpoints')
    if not os.path.isdir('./logs/'+model_name+'/logs'):
        os.makedirs('./logs/'+model_name+'/logs')
    if not os.path.isdir('./logs/'+model_name+'/tensorboard/'+time):
        os.makedirs('./logs/'+model_name+'/tensorboard/'+time)
    if not os.path.isdir('./logs/'+model_name+'/qualitative_results'):
        os.makedirs('./logs/'+model_name+'/qualitative_results/')

    #Custom Callback to save a prediction every epoch
    class prediction_on_epoch(keras.callbacks.Callback):
        def __init__(self, target_size):
            self.target_size = target_size
        def on_epoch_end(self, epoch, logs=None):
            image_to_predict = CITYSCAPES_ROOT+'/leftImg8bit/val/frankfurt/frankfurt_000000_003025_leftImg8bit.png'
            #Get the byte strings for the files
            image_to_predict = tf.io.read_file(image_to_predict)
            #Decode the byte strings as png images
            image_to_predict = tf.image.decode_png(image_to_predict, channels=3)
            #Convert the image to float32 and scale all values between 0 and 1
            image_to_predict = tf.image.convert_image_dtype(image_to_predict, tf.float32)
            #Resize the image to target_size
            image_to_predict = tf.image.resize(image_to_predict, (self.target_size[0], self.target_size[1]), method='nearest')
            #Reshape it ready for prediction as a single batch!
            image_to_predict = tf.expand_dims(image_to_predict, 0)
            #Do a prediction
            prediction = model.predict(image_to_predict)
            #Reshape prediction
            prediction = tf.squeeze(prediction)
            #Resize back upto original size of image
            image_to_save = tf.image.resize(prediction, (1024, 2048), method='nearest')
            #Now do matrix multiply
            image_to_save = np.matmul(prediction, CITYSCAPES_COLORS)
            #Save the Image
            Image.imsave('./logs/{}/qualitative_results/epoch_{}_{}.png'.format(model_name, epoch, model_name), image_to_save.astype(np.uint8), dpi=300)
            


    #The callbacks, has early stopping, logs almost everything, and saves the best weights as the model trains, in theory last weights is best
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        keras.callbacks.CSVLogger('./logs/'+model_name+'/logs/log'+time+'.csv'),
        keras.callbacks.TensorBoard(log_dir='./logs/'+model_name+'/tensorboard/'+time),
        keras.callbacks.ModelCheckpoint('./logs/'+model_name+'/checkpoints/weights-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}-'+time+'.hdf5', 'val_loss', mode='min', save_best_only=True),
        prediction_on_epoch(target_size),
    ]

    #Set the number of steps per epoch
    num_training = len(datasets.get_cityscapes_files(CITYSCAPES_ROOT, 'leftImg8bit', 'train', 'leftImg8bit'))
    num_validation = len(datasets.get_cityscapes_files(CITYSCAPES_ROOT, 'leftImg8bit', 'val', 'leftImg8bit'))
    #Train!
    history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=np.ceil(num_training/batch_size),
            validation_data=val_ds,
            validation_steps=np.ceil(num_validation/batch_size),
            callbacks=callbacks,
            verbose=1,
    )