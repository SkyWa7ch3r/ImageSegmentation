#----------IMPORT PYTHON PACKAGES----------#
import cityscapesscripts.helpers.labels as labels
import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os

#----------IMPORT MODELS AND UTILITIES----------#
import unet, bayes_segnet, deeplabv3plus, fast_scnn
import lrfinder
import datasets

#----------CONSTANTS----------#
CITYSCAPES_LABELS = [label for label in labels.labels if -1 < label.trainId < 255]
#Add unlabeled
CITYSCAPES_LABELS.append(labels.labels[0])
#Get all the colors
CITYSCAPES_COLORS = [label.color for label in CITYSCAPES_LABELS]
#Get the IDS
CITYSCAPES_IDS = [label.trainId for label in CITYSCAPES_LABELS]
#Change unlabeled id from 255 to 19
CITYSCAPES_IDS[-1] = 19
CLASSES=20

#----------ARGUMENTS----------#
parser = argparse.ArgumentParser(prog='train.py', description="Start training a semantic segmentation model")
parser.add_argument("-m", "--model", help="Specify the model you wish to use: OPTIONS: unet, bayes_segnet, deeplabv3+, fastscnn")
parser.add_argument("-r", "--resume", help="Resume the training, specify the weights file path of the format weights-[epoch]-[val-acc]-[val-loss]-[datetime].hdf5")
parser.add_argument('-p',"--path", help="Specify the root folder for cityscapes dataset, if not used looks for CITYSCAPES_DATASET environment variable")
parser.add_argument('-c',"--coarse", help="Use the coarse images", action="store_true")
parser.add_argument('-t',"--target-size", help="Set the image size for training, should be a elements of a tuple x,y,c")
parser.add_argument('-b',"--batch", help="Set the batch size", default=8, type=int)
parser.add_argument('-e',"--epochs", help="Set the number of Epochs", default=1000, type=int)
parser.add_argument('--mixed-precision', help="Use Mixed Precision. WARNING: May cause memory leaks", action="store_true")
parser.add_argument('-f','--lrfinder', help='Use the Learning Rate Finder on a model to determine the best learning rate range for said optimizer')
parser.add_argument('--momentum', help='Only useful for lrfinder, adjusts momentum of an optimizer, if there is that option', type=float, default=0.0)
parser.add_argument('--multi-worker', help='Use MultiWorkerMirroredStrategy Instead of Mirrored Strategy. WARNING: Assumes Use of SLURM due to main use of this on Pawsey Supercomputer Cluster', action='store_true')

models = ['unet', 'bayes_segnet', 'deeplabv3+', 'fastscnn']
optimizers = ['adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', 'rmsprop', 'sgd', 'sgd_nesterov']

args = parser.parse_args()
#Check models
if args.model not in models:
    sys.exit("ERROR: No Model was Chosen or an invalid Model was chosen")
else:
    model_name = args.model
#Check the CITYSCAPES_ROOT path
if os.path.isdir(args.path):
    CITYSCAPES_ROOT = args.path
elif 'CITYSCAPES_DATASET' in os.environ:
    CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_DATASET')
else:
    sys.exit("ERROR: No valid path for Cityscapes Dataset given")
#Now do the target size
if args.target_size is None:
    target_size = (1024, 2048, 3)
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
elif args.resume is not None and not os.path.isfile(args.resume):
    sys.exit("ERROR: Weights File not found.")
if args.lrfinder is not None and args.lrfinder not in optimizers:
    sys.exit("ERROR: An invalid optimizer was chosen")
if args.momentum < 0:
    sys.exit("ERROR: You specified a momentum value and it was below 0")

#----------IMPORT TENSORFLOW AND CONFIGURE IT----------#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

#Set Memory Growth to alleviate memory issues
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

#Set the strategy which will enable Multi-GPU
if args.multi_worker:
    class ModSlurm(tf.distribute.cluster_resolver.SlurmClusterResolver):
            def _resolve_hostnames(self):
                    """
                    If you use shifter or singularity, you cant use scontrol
                    within the container! Thus, a work around was made.
                    While its quick and dirty, it works. Inside an sbatch
                    simply run "scontrol show hostnames > nodes" before running
                    the singularity container.

                    Instead of running the command, it will read a file which has
                    the stdout from the command instead.
                    """
                    #Read the file
                    file = open("nodes","r")
                    #Replace New Line Characters
                    nodes = [node.replace('\n','') for node in file.readlines()]
                    #Return the list of nodes
                    return nodes
    #SLURM GPU values
    if os.environ.get('SLURM_GPUS_PER_NODE') == None:
        gpus_per_node = 1
    else:
        gpus_per_node = int(os.environ.get('SLURM_GPUS_PER_NODE'))
    if os.environ.get('SLURM_GPUS_PER_TASK') == None:
        gpus_per_task = 1
    else:
        gpus_per_task = int(os.environ.get('SLURM_GPUS_PER_TASK'))
    #Create a SlurmClusterResolver
    cluster = ModSlurm({'worker' : int(os.environ.get('SLURM_STEP_NUM_TASKS'))}, gpus_per_node=gpus_per_node, gpus_per_task=gpus_per_task)
    #Create Strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL, cluster_resolver=cluster)
    #Show status of strategy
    print("\nUsing {} GPUs over {} nodes via MultiWorkerMirroredStrategy. Slurm Job {}\n".format(
                                                                                                strategy.num_replicas_in_sync, 
                                                                                                os.environ.get('SLURM_JOB_NUM_NODES'), 
                                                                                                os.environ.get('SLURM_JOB_ID')
                                                                                            )
    )

else:
    #Get Strategy
    strategy = tf.distribute.MirroredStrategy()
    #Show status of Strategy
    print("\nUsing {} GPUs via MirroredStrategy\n".format(strategy.num_replicas_in_sync))

#Change policy to fp16 to use Tensor Cores
if args.mixed_precision:   
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

class csMeanIoU(keras.metrics.Metric):
    """
    Very Similar to the keras Mean IoU 
    with some minor changes specific 
    to this workflow.
    """
    def __init__(self, num_classes, name=None, dtype=None):
        super(csMeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        self.cm = self.add_weight(
            'confusion_matrix',
            shape=(num_classes - 1, num_classes - 1),
            initializer = tf.zeros_initializer(),
            dtype=tf.float64,
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        #Change y_true and y_pred to labels and flatten them
        y_true = tf.argmax(y_true, -1)
        y_pred = tf.argmax(y_pred, -1)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        #Get the Confusion Matrix
        curr_cm = tf.math.confusion_matrix(y_true, y_pred, self.num_classes, dtype=tf.float64)
        #Slice the Confusion Matrix to Ignore the Background
        curr_cm = tf.slice(curr_cm, (0,0), (self.num_classes - 1, self.num_classes - 1))
        #Assign the weights of this to the cm of the class
        return self.cm.assign_add(curr_cm)

    def result(self):
        #Save the cm in a variable for a callback
        confusion_matrix = self.cm
        #Get True Positives, False Positives, False Negatives
        tp = tf.linalg.tensor_diag_part(self.cm)
        fp = tf.math.reduce_sum(self.cm, axis=0) - tp
        fn =  tf.math.reduce_sum(self.cm, axis=1) - tp
        #Calculate the denominator
        denom = tp + fp + fn
        #Get the IoU of each class
        ious = tf.math.divide_no_nan(tp, denom)
        #Get the valid entries (Not 0)
        num_valid_entries = tf.math.reduce_sum(tf.cast(tf.math.not_equal(denom, 0), dtype=tf.float64))
        #Return the Mean IoU of this image and Label
        return tf.math.divide_no_nan(tf.math.reduce_sum(ious), num_valid_entries)

    def reset_states(self):
        keras.backend.set_value(self.cm, np.zeros((self.num_classes-1, self.num_classes-1)))
    
    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(csMeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#With the Mirror Strategy 
with strategy.scope():
    #----------CREATE DATASETS----------#
    if args.multi_worker:
        #Set Batch size to Global Batch Size
        batch_size = batch_size * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    train_ds = datasets.create_dataset(CITYSCAPES_ROOT, batch_size, epochs, target_size, classes=CLASSES)
    val_ds = datasets.create_dataset(CITYSCAPES_ROOT, batch_size, epochs, target_size, train=False, classes=CLASSES)
    #----------CREATE MODEL AND BEGIN TRAINING
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    #Depending on model chosen, get the right model and create the optimizer for it
    if model_name == 'unet':
        model = unet.model(input_size=target_size, num_classes=CLASSES)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    elif model_name == 'bayes_segnet':
        model = bayes_segnet.model(num_classes=CLASSES, input_size=target_size)
        optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=0.001)
    elif model_name == 'fastscnn':
        model = fast_scnn.model(input_size=target_size, num_classes=CLASSES)
        optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=keras.optimizers.schedules.PolynomialDecay(0.045, 1000, power=0.9, end_learning_rate=0.0045))
    elif model_name == 'deeplabv3+':  
        model = deeplabv3plus.model(input_size=target_size, num_classes=CLASSES)
        optimizer = keras.optimizers.SGD(learning_rate=tf.optimizers.schedules.PolynomialDecay(0.007, 1000, power=0.9))
    #We are in LRFINDER mode    
    if args.lrfinder is not None:
        #Set a new optimizer for the LRFINDER
        if args.lrfinder == 'sgd_nesterov':
            optimizer = keras.optimizers.SGD(learning_rate=1e-6, nesterov=True)
        else:
            optimizer = keras.optimizers.get(args.lrfinder)
            optimizer.learning_rate = 1e-6
        if 'momentum' in optimizer.get_config():
            optimizer.momentum = args.momentum
        #Compile the model
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer,  
            metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
        )
        print("Finding Optimum Learning Rate Range")
        #Set the config for the lrfinder
        max_lr = 1e1
        epochs = 5
        num_steps = np.ceil(len(datasets.get_cityscapes_files(CITYSCAPES_ROOT, 'leftImg8bit', 'train', 'leftImg8bit')) / batch_size)
        #Create the learning rate finder
        lrf = lrfinder.LearningRateFinder(model)
        #Start the Finder
        lrf.find(train_ds, 1e-6, 1e1, batch_size, num_steps, epochs)
        #Create title for image, which will also be the name of the image file
        title = model_name + '-' + args.lrfinder 
        if 'momentum' in optimizer.get_config():
            title = title + '-' + str(args.momentum)
        print("The Learning Rate Range has been found")
        #Create the path for the image file
        path = os.path.join('.', 'logs', model_name, 'lrfinder_results', title + '.png')
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print("Plot the Loss")
        #Plot the loss
        lrf.plot_loss(title=title)
        #Save the figure
        plt.savefig(path, dpi=300, format='png')
        print("Please Look Here: {} for your results".format(path))

    #Else we are training a model
    else:
        #Whatever the model is, compile it
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=[
                keras.metrics.CategoricalAccuracy(name='accuracy'), 
                csMeanIoU(num_classes=CLASSES, name="meanIoU")
            ]
        )

        #Ensure directories are made for callbacks
        logs_dir = os.path.join('.','logs',model_name,time)
        csv_dir = os.path.join(logs_dir,'csv')
        images_dir = os.path.join(logs_dir,'images')

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.isdir(csv_dir):
            os.mkdir(csv_dir)
        if not os.path.isdir(images_dir):
            os.mkdir(images_dir)

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
                Image.imsave(images_dir+'/epoch_{}.png'.format(epoch), image_to_save.astype(np.uint8), dpi=300)

        #The callbacks, has early stopping, logs almost everything, and saves the best weights as the model trains, in theory last weights is best
        callbacks=[
            keras.callbacks.CSVLogger(csv_dir+'/log.csv'),
            keras.callbacks.TensorBoard(log_dir=logs_dir),
            keras.callbacks.ModelCheckpoint(logs_dir+'/weights-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}-'+time+'.hdf5', 'val_loss', mode='min', save_best_only=True),
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