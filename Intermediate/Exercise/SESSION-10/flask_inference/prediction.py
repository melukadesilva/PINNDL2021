# # Original code: https://www.paepper.com/blog/posts/pytorch-gpu-inference-with-docker/
import os, io, json, tarfile, glob, time, logging, base64
import numpy as np
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_model():
    model_dir = '../paperspace_gradient/saved_models'
    classes = {
		0: 'Airplane',
		1: 'Autombile',
		2: 'Bird',
		3: 'Cat',
		4: 'Deer',
		5: 'Dog',
		6: 'Frog',
		7: 'Horse',
		8: 'Ship',
		9: 'Truck',
	}

    logger.info(f'Classes are {classes}')    
    model_path = f'{model_dir}/vgg_cifar'
    logger.info(f'Model path is {model_path}')
    model = tf.saved_model.load(model_path)
    return model, classes

# model, classes = load_model()

def predict(model, classes, image_tensor):
    """Predicts the class of an image_tensor."""

    start_time = time.time()
    predict_values = model(image_tensor)
    logger.info("Inference time: {} seconds".format(time.time() - start_time))
    probability_tensor = tf.math.reduce_max(predict_values, 1)[0].numpy()
    index = tf.math.argmax(predict_values, 1)[0].numpy()
    prediction = classes[index]
    probability = "{:1.2f}".format(probability_tensor)
    logger.info(f'Predicted class is {prediction} with a probability of {probability}')
    return {'class': prediction, 'probability': probability}
    
def image_to_tensor(img):
    """Transforms the posted image to a PyTorch Tensor."""
    
    if img.mode == "RGBA":
        # remove the alpha channel
        img = img.convert("RGB")
        
    # reshape the image
    img = np.expand_dims(np.array(img.resize((32, 32))) / 255.0, 0)
    # Remove the alpha channel if exists
    
    # convert to a tensor
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    
    return img_tensor
    
def inference(img):
    """The main inference function which gets passed an image to classify"""

    image_tensor = image_to_tensor(img)
    response = predict(model, classes, image_tensor)
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }


# Function test
model, classes = load_model()

# from PIL import Image

# img = Image.open('../paperspace_gradient/test_data/airplane.png')
# img_tensor = image_to_tensor(img)
# print(img_tensor.shape)

# res = inference(img)
# print(res)

