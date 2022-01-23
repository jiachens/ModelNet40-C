import argparse
import socket
import importlib
import os
import sys
import time
import numpy as np
import tensorflow as tf
from math import sqrt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))

import provider
import data_utils

from data_aug import zoom
from gpu import get_available_gpus, create_parallel_optimization
from mv_utils import PCViews
from mapping2 import OBJECTDATASET_TO_MODELNET, MODELNET_TO_OBJECTDATASET
from utils import RecordExp, get_mv_mean_var

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='multi_res', help='Model name: multi_res')
parser.add_argument('--log_dir', default='./../logs/exp', help='Log dir [default: log]')
parser.add_argument('--file_dir', default='./../logs/exp', help='File dir [default: exp]')
parser.add_argument('--with_bg', default=True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default=True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--no_norm', action="store_true", default=False)
parser.add_argument('--center_data', default=True, help='Whether to explicitly center the data [default: False]')
parser.add_argument('--num_class', type=int, default=15, help='Number of classes to classify.')
parser.add_argument('--train_file',
                    default='./../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5',
                    help='Location of training file')
parser.add_argument('--test_file',
                    default='./../data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5',
                    help='Location of test file')
parser.add_argument('--cross_file',
                    default='./../data/modelnet40_ply_hdf5_2048/test_files.txt',
                    help='Location of test file')

parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 300]')
parser.add_argument('--batch_size', type=int, default=60, help='Batch Size during training [default: 60]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--learning_rate_clip', type=float, default=1e-5, help='Where to clip the lr')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--resolution', type=int, default=512, help='Resolution for image [default: 512]')
parser.add_argument('--size', type=int, default=4, help='Size for points2depth [default: 4]')
parser.add_argument('--trans', type=float, default=-1.4, help='Z-axis translation for point_transform [default: -1.4]')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor [default: 1e-4]')
parser.add_argument('--np2d', action="store_false", default=True,
                    help='Whether to use point2depth or not [default: True]')
parser.add_argument('--aug', action="store_true", default=False,
                    help='Whether to use data augmentation or not [default: False]')
parser.add_argument('--reg_bn', action="store_true", default=False,
                    help='Whether to use regulate bn parameters or not [default: False]')
parser.add_argument('--norm_img', action="store_true", default=False,
                    help='Whether to normalize pictures in p2d or not [default: False]')
parser.add_argument('--nbn', action="store_false", default=True, help='Whether to use bn [default: True]')
parser.add_argument('--resnet_size', type=int, default=18, help='Resnet size [default: 18]')
parser.add_argument('--views', type=int, default=3, help='Num of views [default: 3]')
parser.add_argument('--ratio', type=float, default=0.2, help='Ratio in zoom [default: 0.2]')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma in jitter [default: 0.01]')
parser.add_argument('--clip', type=float, default=0.05, help='Clip in jitter [default: 0.05]')
parser.add_argument('--no_rot_aug', action="store_true", default=False, help='Rotate dataset [default: False]')
parser.add_argument('--visu', action="store_true", default=False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--eval', action="store_true", default=False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--cross_eval', action="store_true", default=False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--no_shuffle', action="store_true", default=False, help="Whether to shuffle the point clouds")
# Resnet parameters
parser.add_argument('--kernel_size', type=int, default=7)
parser.add_argument('--conv_stride', type=int, default=2)
parser.add_argument('--first_pool_size', type=int, default=3)
parser.add_argument('--first_pool_stride', type=int, default=2)
parser.add_argument('--record_file', type=str)

FLAGS = parser.parse_args()

EXP = RecordExp(FLAGS.record_file)
EXP.record_param(vars(FLAGS))
_BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
RESOLUTION = FLAGS.resolution
SIZE = FLAGS.size
TRANS = FLAGS.trans
WEIGHT_DECAY = FLAGS.weight_decay
P2D = FLAGS.np2d
AUG = FLAGS.aug
REG_BN = FLAGS.reg_bn
NORM_IMG = FLAGS.norm_img
VIEWS = FLAGS.views
BN = FLAGS.nbn
RESNET_SIZE = FLAGS.resnet_size
RATIO = FLAGS.ratio
SIGMA = FLAGS.sigma
CLIP = FLAGS.clip
CROSS_FILE = FLAGS.cross_file
EVAL = FLAGS.eval
FILE_DIR = FLAGS.file_dir
# Resnet parameters
KERNEL_SIZE = FLAGS.kernel_size
CONV_STRIDE = FLAGS.conv_stride
FIRST_POOL_SIZE = None if FLAGS.first_pool_size == 0 else FLAGS.first_pool_size
FIRST_POOL_STRIDE = None if FLAGS.first_pool_size == 0 else FLAGS.first_pool_stride
NO_ROT_AUG = FLAGS.no_rot_aug
SHUFFLE = not FLAGS.no_shuffle
CROSS_EVAL = FLAGS.cross_eval

WITH_BG = FLAGS.with_bg
NORMALIZED = FLAGS.norm
if FLAGS.no_norm:
    NORMALIZED = False
TRAIN_FILE = FLAGS.train_file
TEST_FILE = FLAGS.test_file
CENTER_DATA = FLAGS.center_data

IMG_MEAN, IMG_VAR = get_mv_mean_var(
    (
        ('dataset', "modelnet" if "modelnet" in FLAGS.train_file else "object"),
        ('views', VIEWS),
        ('resolution', RESOLUTION),
        ('trans', TRANS),
        ('size', SIZE),
        ('normalize', NORM_IMG),
        ('norm_pc', NORMALIZED),
    )
)
GET_IMG = PCViews().get_img

if VIEWS == 62:
    BATCH_SIZE = _BATCH_SIZE // 6
else:
    BATCH_SIZE = _BATCH_SIZE // VIEWS


MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
os.system('cp ../data_utils.py %s' % (LOG_DIR))  # bkp of data utils
os.system('cp data_aug.py %s' % (LOG_DIR))  # bkp of data aug
os.system('cp multi_model.py %s' % (LOG_DIR))  # bkp of multi model
os.system('cp gpu.py %s' % (LOG_DIR))  # bkp of gpu
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = FLAGS.num_class

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

views = VIEWS
if VIEWS == 62:
    views = 6

print("Number of Classes: " + str(NUM_CLASSES))
print("Normalized: " + str(NORMALIZED))
print("Center Data: " + str(CENTER_DATA))


def get_data_labels(files):
    total_data = np.array([]).reshape((0, 2048, 3))
    total_labels = np.array([]).reshape((0, 1))
    for i in range(len(files)):
        data, labels = data_utils.load_h5(files[i])
        total_data = np.concatenate((total_data, data))
        total_labels = np.concatenate((total_labels, labels))
    total_labels = total_labels.astype(int)
    return total_data, total_labels


MODELNET = True if "modelnet" in TRAIN_FILE else False
###############################################################################################
# Data loading for cross dataset evaluation
if not MODELNET:
    NUM_C = 15
    SHAPE_NAMES = [line.rstrip() for line in open('./shape_names_ext.txt')]
else:
    NUM_C = 40
    SHAPE_NAMES = [line.rstrip() for line in open('./shape_names_modelnet.txt')]

if "modelnet" in CROSS_FILE:
    cross_files = ["../" + line.rstrip() for line in open(CROSS_FILE)]
    CROSS_DATA, CROSS_LABELS = get_data_labels(cross_files)
else:
    if (".h5" in CROSS_FILE):
        CROSS_DATA, CROSS_LABELS = data_utils.load_h5(CROSS_FILE)
    else:
        CROSS_DATA, CROSS_LABELS = data_utils.load_data(CROSS_FILE, NUM_POINT, with_bg_pl=WITH_BG)
##################################################################################################

if "modelnet" in TRAIN_FILE:
    TRAIN_DATA, TRAIN_LABELS = provider.get_modelnet_data(TRAIN_FILE)
    TEST_DATA, TEST_LABELS = provider.get_modelnet_data(TEST_FILE)
else:
    if (".h5" in TRAIN_FILE):
        TRAIN_DATA, TRAIN_LABELS = data_utils.load_h5(TRAIN_FILE)
    else:
        TRAIN_DATA, TRAIN_LABELS = data_utils.load_data(TRAIN_FILE, NUM_POINT, with_bg_pl=WITH_BG)
    if (".h5" in TEST_FILE):
        TEST_DATA, TEST_LABELS = data_utils.load_h5(TEST_FILE)
    else:
        TEST_DATA, TEST_LABELS = data_utils.load_data(TEST_FILE, NUM_POINT, with_bg_pl=WITH_BG)

if (CENTER_DATA):
    TRAIN_DATA = data_utils.center_data(TRAIN_DATA)
    TEST_DATA = data_utils.center_data(TEST_DATA)
    CROSS_DATA = data_utils.center_data(CROSS_DATA)

if (NORMALIZED):
    TRAIN_DATA = data_utils.normalize_data(TRAIN_DATA)
    TEST_DATA = data_utils.normalize_data(TEST_DATA)
    CROSS_DATA = data_utils.normalize_data(CROSS_DATA)

print(len(TRAIN_DATA))
print(len(TEST_DATA))
print(len(CROSS_DATA))

DEVICES = get_available_gpus()


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def log_array(array):
    for i in range(len(array)):
        log_string(str(i) + ' ' + str(array[i]))


log_string('Normalize in p2d: ' + str(NORM_IMG))


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, FLAGS.learning_rate_clip)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def aug(images):
    images = zoom(
        images, BATCH_SIZE * views, RESOLUTION, ratio=RATIO,
        extrapolation_value=((0 - IMG_MEAN) / sqrt(IMG_VAR))
    )
    return images


def transform_to_images(points):
    images = GET_IMG(points, SIZE)
    return images


def loss_filter_fn(v):
    tf.summary.scalar(v.name, tf.nn.l2_loss(tf.cast(v, tf.float32)))
    return True


if REG_BN:
    print('reg_bn')
    loss_filter = loss_filter_fn
else:
    loss_filter = None


# Get model and loss
def training_model(is_training_pl, bn_decay, start, images, labels_pl):
    # Data augmentation
    if AUG:
        images = tf.cond(
            is_training_pl, true_fn=lambda: aug(images), false_fn=lambda: images
        )

    pred, end_points = MODEL.get_model(
        images,
        batch=BATCH_SIZE,
        views=views,
        is_training=is_training_pl,
        num_classes=NUM_CLASSES,
        bn=BN,
        resnet_size=RESNET_SIZE,
        kernel_size=KERNEL_SIZE,
        conv_stride=CONV_STRIDE,
        first_pool_size=FIRST_POOL_SIZE,
        first_pool_stride=FIRST_POOL_STRIDE,
        bn_decay=bn_decay,
    )

    loss = MODEL.get_loss(
        pred, labels_pl, weight_decay=WEIGHT_DECAY,
        end_points=end_points, loss_filter_fn=loss_filter,
        num_classes=NUM_CLASSES
    )
    return loss, pred, start


def train():
    with tf.Graph().as_default():
        images_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, RESOLUTION, views, DEVICES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get training operator
        global_step = tf.train.get_or_create_global_step()
        learning_rate = get_learning_rate(global_step)  # TODO: which step should I use?
        bn_decay = get_bn_decay(global_step)
        tf.summary.scalar('learning_rate', learning_rate)

        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        print(f"DEVICES: {DEVICES}")
        update_op, pred, loss, start = create_parallel_optimization(
            model_fn=training_model,
            devices=DEVICES,
            is_training_pl=is_training_pl,
            bn_decay=bn_decay,
            optimizer=optimizer,
            loss_filter_fn=loss_filter,
            weight_decay=WEIGHT_DECAY,
            controller="/cpu:0",
            images=images_pl,
            labels_pl=labels_pl
        )

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        sess.run(init, {is_training_pl: True})

        ops = {
            'images': images_pl,
            'labels': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss,
            'start': start,
            'train_op': update_op,
            'merged': merged,
            'step': global_step
        }

        best_eval_acc = -1
        best_eval_avg_acc = -1
        best_train_acc = -1
        best_train_eval_acc = -1

        eval_acc, eval_avg_acc, _ = eval_one_epoch(sess, ops, test_writer, test_data=True)
        print(f"Initial Performance: {eval_acc}")
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_acc = train_one_epoch(sess, ops, train_writer)
            eval_acc, eval_avg_acc, _ = eval_one_epoch(sess, ops, test_writer, test_data=True)
            if (epoch % 10) == 9:
                train_eval_acc, train_eval_avg_acc, _ = eval_one_epoch(sess, ops, train_writer, test_data=False)
                if train_eval_acc > best_train_eval_acc:
                    best_train_acc = train_eval_acc

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_eval_avg_acc = eval_avg_acc
                best_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_eval.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            if train_acc > best_train_acc:
                best_train_acc = train_acc

            # Add ops to save and restore all the variables.
            if epoch % 10 == 9:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

        log_string("**** Evaluate Cross **** %s" % epoch)

        EXP.record_result({
            "final_train_acc": train_acc,
            "best_train_acc": best_train_acc,
            "final_train_eval_acc": train_eval_acc,
            "best_train_eval_acc": best_train_eval_acc,
            "best_epoch": best_epoch,
            "final_eval_acc": eval_acc,
            "best_eval_acc": best_eval_acc,
            "best_eval_avg_acc": best_eval_avg_acc
        })

        LOG_FOUT.close()

def evaluate():
    with tf.Graph().as_default():
        images_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, RESOLUTION, views, DEVICES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get training operator
        global_step = tf.train.get_or_create_global_step()
        learning_rate = get_learning_rate(global_step)  # TODO: which step should I use?
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        bn_decay = get_bn_decay(global_step)
        update_op, pred, loss, start = create_parallel_optimization(
            model_fn=training_model,
            devices=DEVICES,
            is_training_pl=is_training_pl,
            bn_decay=bn_decay,
            optimizer=optimizer,
            loss_filter_fn=loss_filter,
            weight_decay=WEIGHT_DECAY,
            controller="/cpu:0",
            images=images_pl,
            labels_pl=labels_pl
        )

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: False})

        # Load checkpoint
        saver.restore(sess, os.path.join(FILE_DIR, 'model.ckpt'))
        log_string("Model restored.")

        ops = {'images': images_pl,
               'labels': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        avg_accuracies, avg_cls_accuracies = [], []
        cls_accuracies = []

        num_evaluations = 10 if SHUFFLE else 1
        for _ in range(num_evaluations):
            if not CROSS_EVAL:
                avg_acc, avg_cls_acc, cls_acc = eval_one_epoch(sess, ops, None)
            else:
                avg_acc, avg_cls_acc, cls_acc = eval_cross(sess, ops, shuffle=True)
            avg_accuracies.append(avg_acc)
            avg_cls_accuracies.append(avg_cls_acc)
            cls_accuracies.append(cls_acc)

        cls_accuracies = np.stack(cls_accuracies)
        for i, name in enumerate(SHAPE_NAMES):
            print('{}: {}'.format(name, cls_accuracies[:, i].tolist()))

        print('accuracies: {}'.format(avg_accuracies))
        print('cls_accuracies: {}'.format(avg_cls_accuracies))

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    current_data, current_label = data_utils.get_current_data_h5(
        TRAIN_DATA, TRAIN_LABELS, NUM_POINT 
    )
    current_label = np.squeeze(current_label)

    start_time = time.time()
    # Augment batched point clouds by rotation and jittering
    if not NO_ROT_AUG:
        rotated_data = provider.rotate_point_cloud(current_data)
    else:
        rotated_data = current_data
    jittered_data = provider.jitter_point_cloud(rotated_data)
    end_time = time.time()
    print(f"Calculating transforms: {end_time - start_time}")

    total_batch_size = BATCH_SIZE * len(DEVICES)
    if len(DEVICES) == 0:
        total_batch_size = BATCH_SIZE
    num_batches = current_data.shape[0] // total_batch_size

    img_total_time = 0
    model_total_time = 0
    for batch_idx in range(num_batches):
        start_time = time.time()

        start_idx = batch_idx * total_batch_size
        end_idx = start_idx + total_batch_size

        images = transform_to_images(jittered_data[start_idx:end_idx])

        images = (images - IMG_MEAN) / sqrt(IMG_VAR)
        # print(np.mean(images), np.var(images))
        end_time = time.time()
        img_total_time += (end_time - start_time)

        feed_dict = {
            ops['images']: images,
            ops['labels']: current_label[start_idx:end_idx],
            ops['is_training_pl']: is_training
        }

        start_time = time.time()
        summary, step, _, loss_val, pred_val, start = sess.run([
            ops['merged'], ops['step'], ops['train_op'],
            ops['loss'], ops['pred'], ops['start']
        ], feed_dict=feed_dict)
        end_time = time.time()
        model_total_time += (end_time - start_time)

        try:
            assert len(pred_val) == len(current_data[start_idx:end_idx])
        except AssertionError:
            print('batch_idx ' + str(batch_idx))
            print('pred ' + str(len(pred_val)))
            print('original ' + str(len(current_data[start_idx:end_idx])))
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += total_batch_size
        loss_sum += loss_val

    print(f"Image time: {img_total_time}")
    print(f"Model time: {model_total_time}")

    acc = (total_correct / float(total_seen))
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % acc)

    return acc

def eval_one_epoch(sess, ops, test_writer, test_data=True):
    """ ops: dict mapping from string to tf ops """

    if test_data:
        current_data, current_label = data_utils.get_current_data_h5(
            TEST_DATA, TEST_LABELS, NUM_POINT
        )
    else:
        print("WARNING: Evaluating on train data")
        current_data, current_label = data_utils.get_current_data_h5(
            TRAIN_DATA, TRAIN_LABELS, NUM_POINT
        )

    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_label = np.squeeze(current_label)
    total_batch_size = BATCH_SIZE * len(DEVICES)
    num_batches = current_data.shape[0] // total_batch_size

    for batch_idx in range(num_batches + 1):
        if batch_idx == num_batches:
            if current_data.shape[0] % total_batch_size == 0:
                pass
            start_idx = current_data.shape[0] - total_batch_size
            end_idx = current_data.shape[0]
        else:
            start_idx = batch_idx * total_batch_size
            end_idx = (batch_idx + 1) * total_batch_size

        images = transform_to_images(current_data[start_idx:end_idx])

        images = (images - IMG_MEAN) / sqrt(IMG_VAR)

        feed_dict = {ops['images']: images,
                     ops['labels']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}

        if test_writer is not None:
            summary, step, loss_val, pred_val, start = sess.run([
                ops['merged'], ops['step'], ops['loss'], ops['pred'], ops['start']
            ], feed_dict=feed_dict)
        else:
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

        try:
            assert len(pred_val) == len(current_data[start_idx:end_idx])
        except AssertionError:
            print('pred ' + str(len(pred_val)))
            print('original ' + str(len(current_data[start_idx:end_idx])))

        if test_writer is not None:
            test_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 1)
        if batch_idx == num_batches:
            start_idx = num_batches * total_batch_size
            current_start = total_batch_size - current_data.shape[0] % total_batch_size
            try:
                assert pred_val[current_start:].shape[0] == end_idx - start_idx
            except AssertionError:
                log_string('start_index: ' + start_idx)
            pred_val = pred_val[current_start:]
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += (end_idx - start_idx)
            loss_sum += loss_val * (end_idx - start_idx)
        else:
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += total_batch_size
            loss_sum += (loss_val * total_batch_size)
        for i in range(start_idx, end_idx):
            label = current_label[i]
            total_seen_class[label] += 1
            total_correct_class[label] += (pred_val[i - start_idx] == label)

    eval_acc = (total_correct / float(total_seen))
    eval_cls_acc = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    eval_avg_acc = (np.mean(eval_cls_acc))

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % eval_acc)
    log_string('eval avg class acc: %f' % eval_avg_acc)

    for i, name in enumerate(SHAPE_NAMES):
        if (total_seen_class[i] == 0):
            accuracy = -1
        else:
            accuracy = total_correct_class[i] / float(total_seen_class[i])
        log_string('%10s:\t%0.3f' % (name, accuracy))

    return eval_acc, eval_avg_acc, eval_cls_acc


def eval_cross(sess, ops, num_votes=1, topk=1, shuffle=False):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    total_seen_class    = [0 for _ in range(NUM_C)]
    total_correct_class = [0 for _ in range(NUM_C)]
    truth_prediction    = [[0 for _ in range(NUM_C)] for _ in range(NUM_C)]
    fout = open(os.path.join(LOG_DIR, 'pred_label.txt'), 'w')

    current_data, current_label = data_utils.get_current_data_h5(
        CROSS_DATA, CROSS_LABELS, NUM_POINT 
    )
    current_label = np.squeeze(current_label)

    ####################################################
    print(current_data.shape)
    print(current_label.shape)

    filtered_data = []
    filtered_label = []
    for i in range(current_label.shape[0]):
        modelnet = False if "modelnet" in CROSS_FILE else True
        diction = OBJECTDATASET_TO_MODELNET if modelnet else MODELNET_TO_OBJECTDATASET
        if (current_label[i] in diction.keys()):
            filtered_label.append(current_label[i])
            filtered_data.append(current_data[i, :])

    filtered_data = np.array(filtered_data)
    filtered_label = np.array(filtered_label)
    print(filtered_data.shape)
    print(filtered_label.shape)

    current_data = filtered_data
    current_label = filtered_label
    ###################################################

    total_batch_size = BATCH_SIZE * len(DEVICES)
    num_batches = current_data.shape[0] // total_batch_size
    for batch_idx in range(num_batches + 1):
        if batch_idx == num_batches:
            if current_data.shape[0] % total_batch_size == 0:
                pass
            start_idx = current_data.shape[0] - total_batch_size
            end_idx = current_data.shape[0]
        else:
            start_idx = batch_idx * total_batch_size
            end_idx = (batch_idx + 1) * total_batch_size
        cur_batch_size = end_idx - start_idx
        if batch_idx == num_batches:
            cur_batch_size = end_idx - num_batches * total_batch_size

        # Aggregating BEG
        batch_loss_sum = 0  # sum of losses for the batch
        num = 15 if not modelnet else 40
        batch_pred_sum = np.zeros((cur_batch_size, num))  # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, num))  # 0/1 for classes
        for vote_idx in range(num_votes):
            images = transform_to_images(current_data[start_idx:end_idx])
            images = (images - IMG_MEAN) / sqrt(IMG_VAR)
            feed_dict = {
                ops['images']: images,
                ops['labels']: current_label[start_idx:end_idx],
                ops['is_training_pl']: is_training
            }
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

            if batch_idx == num_batches:
                start_idx = num_batches * total_batch_size
                current_start = total_batch_size - current_data.shape[0] % total_batch_size
                try:
                    assert pred_val[current_start:].shape[0] == end_idx - start_idx
                except AssertionError:
                    log_string('start_index: ' + start_idx)
                pred_val = pred_val[current_start:]

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        pred_val = np.argmax(batch_pred_sum, 1)
        # Aggregating END

        for i in range(start_idx, end_idx):
            if modelnet:
                total_seen += 1
                if (pred_val[i - start_idx] not in MODELNET_TO_OBJECTDATASET.keys()):
                    continue
                pred = MODELNET_TO_OBJECTDATASET[pred_val[i - start_idx]]
                if (pred == current_label[i]):
                    total_correct += 1
            else:
                total_seen += 1
                if (pred_val[i - start_idx] not in OBJECTDATASET_TO_MODELNET.keys()):
                    continue
                else:
                    possible_label = OBJECTDATASET_TO_MODELNET[pred_val[i - start_idx]]
                    if (current_label[i] in possible_label):
                        total_correct += 1

        for i in range(start_idx, end_idx):
            if modelnet:
                label = current_label[i]
                total_seen_class[label] += 1

                if pred_val[i - start_idx] not in MODELNET_TO_OBJECTDATASET:
                    pred_label = "NA"
                else:
                    pred = MODELNET_TO_OBJECTDATASET[pred_val[i - start_idx]]
                    total_correct_class[label] += (pred == label)
                    truth_prediction[label][pred] += 1

                    pred_label = SHAPE_NAMES[pred]

                groundtruth_label = SHAPE_NAMES[label]
            else:
                label = current_label[i]
                total_seen_class[diction[label]] += 1

                if (pred_val[i - start_idx] in OBJECTDATASET_TO_MODELNET.keys()):
                    possible_label = OBJECTDATASET_TO_MODELNET[pred_val[i - start_idx]]
                    if (label in possible_label):
                        total_correct_class[MODELNET_TO_OBJECTDATASET[label]] += 1
                        truth_prediction[MODELNET_TO_OBJECTDATASET[label]][pred_val[i - start_idx]] += 1

                pred_label = SHAPE_NAMES[pred_val[i - start_idx]]
                groundtruth_label = SHAPE_NAMES[MODELNET_TO_OBJECTDATASET[label]]

            fout.write('%s, %s\n' % (pred_label, groundtruth_label))

            if pred_val[i - start_idx] != label and FLAGS.visu:  # ERROR CASE, DUMP!
                # save ply
                ply_filename = '%d_label_%s_pred_%s.ply' % (error_cnt, groundtruth_label, pred_label)
                data_utils.save_ply(np.squeeze(current_data[i, :, :]), ply_filename)
                error_cnt += 1

    log_string('total seen: %d' % (total_seen))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    seen_class_accuracies = []
    seen_correct_class = []
    for i in range(len(total_seen_class)):
        if total_seen_class[i] != 0:
            seen_class_accuracies.append(total_seen_class[i])
            seen_correct_class.append(total_correct_class[i])
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(seen_correct_class) / np.array(seen_class_accuracies, dtype=np.float))))

    seen_correct_class = np.array(seen_correct_class)
    seen_class_accuracies = np.array(seen_class_accuracies)

    for i, name in enumerate(SHAPE_NAMES):
        if (total_seen_class[i] == 0):
            accuracy = -1
        else:
            accuracy = total_correct_class[i] / float(total_seen_class[i])
        log_string('%10s:\t%0.3f' % (name, accuracy))

    avg_acc = total_correct / float(total_seen)
    cls_avg_acc = np.mean(seen_correct_class / seen_class_accuracies)

    total_correct_class = np.array(total_correct_class)
    total_seen_class    = np.array(total_seen_class)
    unseen_class        = (total_seen_class == 0)
    total_correct_class[unseen_class] = -1
    total_seen_class[unseen_class]    = 1

    return avg_acc, cls_avg_acc, total_correct_class / total_seen_class


if __name__ == "__main__":
    if EVAL:
        evaluate()
    else:
        train()
