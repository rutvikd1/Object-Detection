import argparse
import io
import os
import subprocess

import ray
import tensorflow.compat.v1 as tf
from PIL import Image
from psutil import cpu_count
from nuimages import NuImages
import numpy as np
# from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import (get_module_logger, int64_feature, int64_list_feature,
                bytes_list_feature, bytes_feature, float_list_feature)   #parse_frame,


NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}

NUS_CATEGORIES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

NUS_CATEGORIES_TO_TAKE = ('car', 'truck', 'bicycle', 'pedestrian')



def create_tf_example(nuim, filename, encoded_jpeg, token, resize=True): #(filename, encoded_jpeg, annotations, resize=True):
    """
    This function create a tf.train.Example from the nuimage frame.

    args:
        - filename [str]: name of the image
        - encoded_jpeg [bytes]: jpeg encoded image

    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """
    data_dir = nuim.dataroot
    if not resize:
        encoded_jpg_io = io.BytesIO(encoded_jpeg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        width_factor, height_factor = image.size
    else:
        print(f"filename is {filename}")
        image_tensor = tf.io.decode_jpeg(encoded_jpeg)
        height_img_original, width_img_original, _ = image_tensor.shape
        image_res = tf.cast(tf.image.resize(image_tensor, (640, 640)), tf.uint8)
        encoded_jpeg = tf.io.encode_jpeg(image_res).numpy()
        width, height = 640, 640
        height_factor = 640/ height_img_original
        width_factor = 640/ width_img_original

    mapping = { 'car':1,'truck':2 , 'bicycle':3, 'pedestrian':4}
    image_format = b'jpg'
    xmins = list()
    xmaxs = list()
    ymins = list()
    ymaxs = list()
    classes_text = list()
    classes = list()
    boxes = list()
    labels = list()
    filename = filename.encode('utf8')

    bboxes_tokens,_ = nuim.list_anns(nuim.get('sample_data',token)['sample_token'])

    for j in range(len(bboxes_tokens)):
            
        bboxes = nuim.get('object_ann', bboxes_tokens[j])['bbox']
        # print(f"bboxes are {bboxes}\n and the elements are {bboxes[0]}")
        # order = [1,0,3,2]   
        # scaled_boxes = [bboxes[i] for i in order]
        bboxes[1] *= height_factor  # ymin
        bboxes[3] *= height_factor  # ymax
        bboxes[0] *= width_factor  # xmin
        bboxes[2] *= width_factor  # xmax
        
        label = nuim.get('category', nuim.get('object_ann',bboxes_tokens[j])["category_token"])['name']
        nus_cat = NAME_MAPPING.get(label)
        print(f"label is {label} and nus_cat is {nus_cat}")
        if nus_cat in NUS_CATEGORIES_TO_TAKE:

            boxes.append(bboxes)
            labels.append(label)
            xmins.append(bboxes[0])
            ymins.append(bboxes[1])
            xmaxs.append(bboxes[2])
            ymaxs.append(bboxes[3])
            classes.append(mapping[nus_cat])
            classes_text.append(nus_cat.encode('utf8'))
    # return boxes, labels


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def process_tfr(nuim,token,path):
    """
    process a nuimage record into a tf api tf record

    args:
        - path [str]: path to the nuimage file
        - data_dir [str]: path to the destination directory
    """
    # create processed data dir
    data_dir = nuim.dataroot
    dest = os.path.join(data_dir, camera + '_processed')
    
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
    
    file_name = path #os.path.basename(path)
    print(f"the filename in process_tfr is {file_name}")
    logger.info(f'Processing {path}')

    filename = os.path.join(dest , os.path.basename(file_name).replace('.jpg','.tfrecords'))
    writer = tf.python_io.TFRecordWriter(filename)
    dataset = tf.data.TFRecordDataset(path, compression_type='')

    # for idx, data in enumerate(dataset):
    #     # we are only saving every 10 frames to reduce the number of similar
    #     # images. Remove this line if you have enough space to work with full
    #     # temporal resolution data.
    #     if idx % 3 == 0:
    #         frame = open_dataset.Frame()
    
    # frame.ParseFromString(bytearray(data.numpy()))
    # encoded_jpeg, annotations = parse_frame(frame)

    # filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
    encoded_jpeg = open(path, 'rb').read()
    # annotations = nuim.get('sample_data', token)['annotations']
    
    tf_example = create_tf_example(nuim, file_name, encoded_jpeg, token)#, annotations)
    writer.write(tf_example.SerializeToString())
    writer.close()



def get_file_name(nuim,token):
    """
    select path of a single image for processing
    args:
        - nuim [NuImages]: nuimages object
        - token [str]: token of the sample data
    
    returns:
        - filename [str]: name of the file
    """
    
    # create data dir
    data_dir = nuim.dataroot
    dest = os.path.join(data_dir, 'samples/'+ camera)
    
    if not os.path.exists(dest):
        logger.log(f'Camera {camera} not found.')

    local_path = dest + '/' + os.path.basename(nuim.get('sample_data',token)['filename'])
    return local_path


# @ray.remote
def download_and_process(nuim, token):
    logger = get_module_logger(__name__)
    # need to re-import the logger because of multiprocesing
    local_path = get_file_name(nuim,token)
    process_tfr(nuim, token,local_path)
    # remove the original tf record to save space
    logger.info(f'Deleting {local_path}')
    # os.remove(local_path)


def get_sample_data_tokens(nuim, camera):
    """
    Get the tokens for a specific camera

    args:
        - nuim [NuImages]: nuimages object
        - camera [str]: camera name (e.g. CAM_FRONT)

    returns:
        - tokens [list]: list of tokens for the camera
    """
    front_sample_data_tokens = []

    for j in range(len(nuim.sample_data)):
        curr_sample_data_token = nuim.sample_data[j]['token']
        if nuim.shortcut('sample_data', 'sensor',curr_sample_data_token)['channel'] == camera and nuim.sample_data[j]['is_key_frame'] == True:
            front_sample_data_tokens.append(curr_sample_data_token)
    return front_sample_data_tokens



if __name__ == "__main__":
    """
    parse args
    this is a simple script to download and process the nuimages dataset
    into tf records. It uses the ray library to parallelize the download and
    processing of the files. The script takes the following arguments:
    - data_dir: path to the nuimages dataset
    - version: version of the dataset (e.g. v1.0-mini)
    - verbose: verbose mode
    - lazy: lazy mode
    - camera: camera name (e.g. CAM_FRONT)
    """
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process files to tf records')

    parser.add_argument('--data_dir', required=True, default='/data/sets/nuimages',type=str,
                        help='data directory')
    
    parser.add_argument('--version', required=False, default='v1.0-mini', type=str,
                        help='Version of the dataset (e.g. v1.0-mini)')
    
    parser.add_argument('--verbose', required=False, default=True, type=bool,
                        help='Verbose mode')
    
    parser.add_argument('--lazy', required=False, default=True, type=bool,
                        help='Lazy mode')
    
    parser.add_argument('--camera', required=False, default='CAM_FRONT', type=str,
                        help='Camera name (e.g. CAM_FRONT)')
    
    parser.add_argument('--size', required=False, default=8, type=int,
                        help='Number of images to process')
        
    args = parser.parse_args()
    data_dir = args.data_dir
    size = args.size
    camera = args.camera

    nuim = NuImages(dataroot=args.data_dir, version=args.version, verbose=True, lazy=True)

    tokens = get_sample_data_tokens(nuim, camera=camera)

    logger.info(f' {len(tokens[:size])} images to process. Be patient, this will take a long time.')

    for k in range(size):
      download_and_process(nuim,tokens[k])

    # init ray
    # ray.init(num_cpus=cpu_count())
    # workers = [download_and_process.remote(nuim, fn) for fn in tokens[:size]]
    # _ = ray.get(workers)