import keras2onnx
import tensorflow as tf
from tensorflow import keras
import os, argparse
import fast_scnn, deeplabv3plus, separable_unet

def keras_to_onnx(model_choice, weights):
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
    model.load_weights(weights)
    # Plot the model for visual purposes in case anyone asks what you used
    tf.keras.utils.plot_model(
        model, to_file=os.path.join('./results', model_choice, model_choice+'.png'), show_shapes=True, dpi=300)

    onnx_model = keras2onnx.convert_keras(model, model.name)

    onnx_filename = os.path.join('results', model_choice,model_choice+ '_' + input_size + '.onnx')
    
    keras2onnx.save_model(onnx_model, onnx_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to convert to uff",
                        choices=['fastscnn', 'deeplabv3+', 'separable_unet'])
    parser.add_argument("-w", "--weights-file", type=str, default="",
                        help="The weights the model will use in the uff file")
    args = parser.parse_args()
    if os.path.isfile(args.weights_file):
        keras_to_onnx(args.model, args.weights_file)
    else:
        parser.error("Please provide a weights file. File given not found.")