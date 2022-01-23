import tensorflow as tf
import sys
import os
import tf_util
import multi_model as resnet_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))


def placeholder_inputs(batch_size, num_point, resolution, views, devices):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size * views * len(devices), resolution, resolution, 1))
    labels_pl = tf.placeholder(tf.int32, shape=batch_size * len(devices))
    return pointclouds_pl, labels_pl

def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
        resnet_size: The number of convolutional layers needed in the model.
    Returns:
        A list of block sizes to use in building the model.
    Raises:
        KeyError: if invalid resnet_size is received.
    """
    choices = {
        9: [2, 2],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = (
            'Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys())
        )
    raise ValueError(err)


def get_model(images, batch, views, is_training, bn_decay, num_classes=15, bn=True, resnet_size=18, kernel_size=7,
              conv_stride=2, first_pool_size=3, first_pool_stride=2):
    """ Classification Resnet, input is (BxV)XRXRX1, output Bx40 """
    resnet_version = 2
    data_format = None
    dtype = resnet_model.DEFAULT_DTYPE
    print(resnet_size)
    block_strides = [1, 2] if resnet_size == 9 else [1, 2, 2, 2]
    model = resnet_model.Model(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        first_pool_size=first_pool_size,
        first_pool_stride=first_pool_stride,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=block_strides,
        bn_decay=bn_decay,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype,
        bn=bn
    )

    print(f"kernel_size:       {kernel_size}")
    print(f"conv_stride:       {conv_stride}")
    print(f"first_pool_size:   {first_pool_size}")
    print(f"first_pool_stride: {first_pool_stride}")

    features = model(images, training=is_training)  # (BXV)XF
    with tf.compat.v1.variable_scope("extract", reuse=tf.compat.v1.AUTO_REUSE):
        if views != 1:
            out = tf.reshape(features, [batch * views, -1, 1, 1])
            out = tf_util.batch_norm_for_conv2d(out, is_training, bn_decay, scope="bn1")
            out = tf.reshape(out, [batch, views, -1])
            out = tf_util.dropout(out, is_training, scope="dp1")
            out = tf.reshape(out, [batch, -1])
            out = tf_util.fully_connected(out, 128, scope="fc1", is_training=is_training, bn=bn, bn_decay=bn_decay)
            out = tf_util.dropout(out, is_training, scope="dp2")
        else:
            out = features
        out = tf_util.fully_connected(out, num_classes, activation_fn=None, scope='fc3')  # BXnum_classes
    end_points = {}
    return out, end_points

def get_loss(pred, label, weight_decay, end_points, loss_filter_fn=None, num_classes=15):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(v):
        tf.summary.scalar(v.name, tf.nn.l2_loss(tf.cast(v, tf.float32)))
        return 'batch_normalization' not in v.name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm
    print(loss_filter_fn)

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.compat.v1.trainable_variables()
            if loss_filter_fn(v)
        ])

    tf.summary.scalar('l2_loss', l2_loss)
    loss = classify_loss + l2_loss
    return loss

def parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    return total_parameters


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
