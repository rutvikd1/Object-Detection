import logging

import tensorflow.compat.v1 as tf
from object_detection.inputs import train_input
from object_detection.utils.label_map_util import create_category_index_from_labelmap
import pandas as pd
from object_detection.utils.config_util import get_configs_from_pipeline_file


def get_dataset(tfrecord_path, label_map='label_map.pbtxt'):
    """
    Opens a tf record file and create tf dataset
    args:
      - tfrecord_path [str]: path to a tf record file
      - label_map [str]: path the label_map file
    returns:
      - dataset [tf.Dataset]: tensorflow dataset
    """
    tf_rec_files = tf.io.gfile.glob(tfrecord_path)
    num_parallel_reads=tf.data.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tf_rec_files, num_parallel_reads=num_parallel_reads)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def parse_function(example):
    features = {
        # 'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, features)
    return parsed_example

def parse_tfrecord_fn(example):
    
    
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    width = tf.cast(example['image/width'], tf.float32)
    height = tf.cast(example['image/height'], tf.float32)
    # height = tf.sparse.to_dense(example['image/height'])
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])*width
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])*width
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])*height
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])*height
    labels = tf.sparse.to_dense(example['image/object/class/label'])

    boxes = tf.stack([ymin,xmin,ymax, xmax], axis=1)
    return image, boxes, labels

def get_module_logger(mod_name):
    """ simple logger """
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_train_input(config_path):
  """
  Get the tf dataset that inputs training batches
  args:
    - config_path [str]: path to the edited config file
  returns:
    - dataset [tf.Dataset]: data outputting augmented batches
  """
  # parse config
  configs = get_configs_from_pipeline_file(config_path)
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  # get the dataset
  dataset = train_input(train_config, train_input_config, configs['model'])
  return dataset

def dataset_to_dataframe(dataset):
  columns = ['filename', 'boxes','xmin', 'ymin', 'xmax', 'ymax','box_width', 'box_height',  'area', 'aspect_r' , 'labels']
  category_index = {1 : 'car' , 2:  'truck' , 3 : 'bicycle', 4: 'pedestrian' }


  data = pd.DataFrame(columns=columns) 
  # try:
  for j, raw_record in enumerate(dataset):
      # print(f"processing {j}th record")
      image_tensor, boxes_tensor, labels_tensor = parse_tfrecord_fn(raw_record)
      boxes = boxes_tensor.numpy()
      labels = labels_tensor.numpy()
      width = raw_record['image/width'].numpy()
      height = raw_record['image/height'].numpy()
      for box, label in zip(boxes, labels):
        
        ymin, xmin, ymax, xmax = box
        ymin = ymin/height
        xmin = xmin/width
        ymax = ymax/height
        xmax = xmax/width
        box_width = xmax - xmin
        box_height = ymax - ymin
        area = box_width * box_height
        aspect_r = box_width / box_height
        
        data.loc[len(data)] = {
             "filename": raw_record['image/filename'].numpy().decode('utf-8'),
             "boxes": box,
             "xmin": xmin,
             "ymin": ymin,
             "xmax": xmax,
             "ymax": ymax,
             "area": area,
             "box_width": box_width,
             "box_height": box_height,
             "aspect_r": aspect_r,
             "labels": category_index[label]
         }

  # except 'UnknownError':
  #   print("this one had a problem")
  return data


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))