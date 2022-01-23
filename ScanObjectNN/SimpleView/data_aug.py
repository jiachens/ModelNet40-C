import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 1))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row * 32:(row + 1) * 32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()

def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def jitter(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter images. jittering is per pixel.
        Input:
          BxNxNx1 array, original batch of images
        Return:
          BxNxNx1 array, jittered batch of images
    """
    assert(clip > 0)
    mask = tf.equal(batch_data, 0.0)
    r = tf.clip_by_value(sigma * tf.random.normal(tf.shape(batch_data), dtype=tf.float32), -1 * clip, clip)
    add = tf.add_n([batch_data, r])
    targets = tf.where(mask, batch_data, add)
    return targets

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor, batch: int, resolution: int, ratio=0.2, extrapolation_value=0) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    if extrapolation_value == 0:
        print("WARNING: using 0 for the extrapolated value")

    boxes = np.concatenate((np.zeros((batch, 2)), np.ones((batch, 2))), axis=1)
    ind = np.arange(0, batch, 1)
    num = tf.random.uniform(shape=[batch, 4], minval=-ratio, maxval=ratio, dtype=tf.dtypes.float32)
    boxes = tf.add(tf.constant(boxes, dtype=tf.float32), num)
    return tf.image.crop_and_resize(
        x, boxes=boxes, box_ind=ind, crop_size=(resolution, resolution), method='nearest',
        extrapolation_value=extrapolation_value
    )
