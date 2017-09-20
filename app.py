"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes Caffe to perform inference with this image
    - Returns the output

"""
import os
import sys
import caffe
import numpy
import logging

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from PIL import Image

# config
MODEL_WEIGHTS_LOCATION = '/weights/alexnet.caffemodel'
MODEL_DEFINITION_LOCATION = './alexnet.prototxt'
MEAN_VALUE_BLUE = 296
MEAN_VALUE_GREEN = 103
MEAN_VALUE_RED = 108
MEAN_ARRAY = numpy.array([MEAN_VALUE_RED, MEAN_VALUE_GREEN, MEAN_VALUE_BLUE])
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# checks
def check_file(path, message):
  if os.path.isfile(path):
    logger.info(message + ' found.')
  else:
    logger.error(message + ' not found! Exiting...')
    quit()


check_file(MODEL_WEIGHTS_LOCATION, 'Caffe model weights')
check_file(MODEL_DEFINITION_LOCATION, 'Caffe model definition')

# caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
net = caffe.Net(MODEL_DEFINITION_LOCATION, MODEL_WEIGHTS_LOCATION, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', MEAN_ARRAY)  # subtract the mean
transformer.set_raw_scale('data', 255)  # rescale from [0, 255] to [0, 1]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          159, 240)  # image size is 240x159

app = Flask(__name__)


@app.route('/', methods=["POST"])
def assess_photo(path):
  """
  Take the input image and assess it
  """
  logger.info("API called!")
  # check if the post request has the file part
  if 'file' not in request.files:
    return BadRequest("File not present in request")
  file = request.files['file']
  if file.filename == '':
    return BadRequest("File name is not present in request")
  if not allowed_file(file.filename):
    return BadRequest("Invalid file type")
  if file and allowed_file(file.filename):
    logger.info("Received a photo!")
    image = Image.open(file)
    logger.info("Photo resolution is {}x{} pixels.".format(image.size[0], image.size[1]))

    return get_photo_score(image)


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# does inference in Caffe and returns the photo score
def get_photo_score(photo):
  if photo.size[0] < photo.size[1]:
    logger.info("Image is horizontal - rotating to vertical...")
  try:
    photo = photo.rotate(90)
  except OSError as error:
    logger.error(error.strerror)

  logger.info("Resizing the image to 240x159 pixels...")
  photo = photo.resize((240, 159), 3)

  logger.info("Loading image to Caffe...")
  photo.save("./temp.jpg")
  photo = caffe.io.load_image("./temp.jpg")
  os.remove("./temp.jpg")
  transformed_image = transformer.preprocess('data', photo)
  net.blobs['data'].data[...] = transformed_image

  logger.info("Doing the forward propagation...")
  output = net.forward()

  score = output['prob'][0][1]  # output, first image, second neuron
  score = round(float(score) * 100, 2)
  logger.info("Done. Photo score is {}%.".format(score))
  return jsonify(score=score)


if __name__ == '__main__':
  app.run(host='0.0.0.0')
  logger.info("Server launched!")
  logger.info("Waiting for requests...")
