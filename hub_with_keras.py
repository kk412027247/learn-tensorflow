from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL.Image as Image
from setproxy import setproxy

setproxy()

classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

grace_hopper = tf.keras.utils.get_file(
    'image.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
# grace_hopper.show()

grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape)

result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title('Prediction: ' + predicted_class_name.title())
plt.show()

data_root = tf.keras.utils.get_file(
    'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

print('image_data', image_data)

for image_batch, label_batch in image_data:
    print('IMAGE batch shape: ', image_batch.shape)
    print('LABEL batch shape: ', label_batch.shape)
    break

result_batch = classifier.predict(image_batch)
result_batch.shape
