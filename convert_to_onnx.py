import keras2onnx
import tensorflow as tf
from tensorflow import keras
import os, argparse
import fast_scnn, deeplabv3plus, separable_unet
import onnx

def keras_to_onnx(model_choice, weights):
    """
    If this is being run on the Jetson
    Then Tensorflow 1.15.0 is recommended,
    keras2onnx 1.6 and onnxconverter-runtime 1.6 installed via pip.
    Its able to be converted.
    """
    if model_choice == 'fastscnn':
        model = fast_scnn.model(num_classes=20, input_size=(1024, 2048, 3))
        input_size = '1024x2048'
    elif model_choice == 'deeplabv3+':
        # Its important here to set the output stride to 8 for inferencing
        model = deeplabv3plus.model(num_classes=20, input_size=(
            1024, 2048, 3), depthwise=True, output_stride=8)
        input_size = '1024x2048'
    elif model_choice == 'separable_unet':
        # Was trained on a lower resolution
        model = separable_unet.model(num_classes=20, input_size=(512, 1024, 3))
        input_size = '512x1024'
    # Whatever the model is, load the weights chosen
    print("Loading the weights for {} at input size {}".format(model_choice, input_size))
    model.load_weights(weights)
    print("Weights Loaded")
    # Plot the model for visual purposes in case anyone asks what you used
    print("Plotting the model")
    tf.keras.utils.plot_model(
        model, to_file=os.path.join('./results', model_choice, model_choice+'.png'), show_shapes=True, dpi=300)
    #Convert keras model to onnx
    print("Converting Keras Model to ONNX")
    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=8)
    #Get the filename
    onnx_filename = os.path.join('results', model_choice,model_choice+ '_' + input_size + '.onnx')
    #Save the ONNX model
    print("Saving the ONNX Model")
    onnx.save_model(onnx_model, onnx_filename)
    print("Conversion Complete, ready for Jetson AGX Xavier")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to convert to uff",
                        choices=['fastscnn', 'deeplabv3+', 'separable_unet'])
    parser.add_argument("-w", "--weights-file", type=str, default="",
                        help="The weights the model will use in the uff file")
    args = parser.parse_args()
    #If its a file then use the keras_to_onnx
    if os.path.isfile(args.weights_file):
        keras_to_onnx(args.model, args.weights_file)
    #Else give an error
    else:
        parser.error("Please provide a weights file. File given not found.")