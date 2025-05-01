import argparse
import time
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
from matplotlib import animation
import glob
from PIL import Image
from utils import get_module_logger

tf.get_logger().setLevel('ERROR')


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    print(f"path is {path}")
    return np.array(Image.open(path))

def main(model_path, label_map_path, image_paths, output_path):
    # Load the model
    start_time = time.time()
    print("Loading model...")
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds, successfully.")
    
    # Load the label map
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    print("Label map loaded successfully.")

    images = list()
    for idx, img_path in enumerate(image_paths):

        image_np = load_image_into_numpy_array(img_path)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        img_w_detections = \
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.5,
            agnostic_mode=False)
        
        images.append(image_np_with_detections)
        
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.subplot(111)
    ax.axis('off')
    im_obj = ax.imshow(images[0])

    def animate(idx):
        image = images[idx]
        im_obj.set_data(image)

    anim = animation.FuncAnimation(fig, animate, frames=len(images))
    anim.save(output_path, fps=3, dpi=300)
    
if __name__ == "__main__":
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description='Make a gif with the detections')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model directory')
    parser.add_argument('--label_map_path', type=str, required=True,
                        help='Path to the label map file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output gif file')
    parser.add_argument('--output_file_name', type=str, required=False,
                        default='output.gif',
                        help='Name of the output gif file (optional)')
    args = parser.parse_args()
    main(args.model_path, args.label_map_path, glob.glob(args.image_path + '/*.jpg'), args.output_path, args.output_file_name)
