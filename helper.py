import re
import random
import numpy as np
import os.path
import scipy
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = cv2.resize(cv2.imread(image_file).astype('uint8'), image_shape[::-1])
                gt_image = cv2.resize(cv2.imread(gt_image_file).astype('uint8'), image_shape[::-1])

                image_a, gt_image_a = augment(image, gt_image)

                #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                #gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image_a == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image_a)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def get_pair(data_folder="./data/data_road/training", image_shape=(160, 576)):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    image_path = random.choice(image_paths)
    print(image_path)
    label_path = label_paths[os.path.basename(image_path)]
    image = cv2.resize(cv2.imread(image_path).astype('uint8'), image_shape[::-1])
    label_img = cv2.resize(cv2.imread(label_path).astype('uint8'), image_shape[::-1])
    return image, label_img

##
## Image augmentation
##

def bgr2rgb(img):
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img2

def bgr2hls(img):
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    return img2

def hls2bgr(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    return img1

def bgr2hsv(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img1

def hsv2bgr(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img1

def add_random_shadow(image):
    #
    # Add a random shadow to a BGR image to pretend
    # we've got clouds or other interference on the road.
    #
    rows,cols,_ = image.shape
    top_y = cols*np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols*np.random.uniform()
    image_hls = bgr2hls(image)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y)-(bot_x - top_x)*(Y_m-top_y) >=0)] = 1
    random_bright = .25+.7*np.random.uniform()
    if (np.random.randint(2) ==1 ):
        random_bright = .5
        cond1 = (shadow_mask==1)
        cond0 = (shadow_mask==0)
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = hls2bgr(image_hls)
    return image

def augment_brightness_camera_images(image):
    #
    # expects input image as BGR, adjusts brightness to
    # pretend we're in different lighting conditions.
    #
    image1 = bgr2hsv(image)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image2 = hsv2bgr(image1)
    return image2

def trans_image(img,label,trans_range):
    #
    # Shift image up or down a bit within trans_range pixels,
    # filling missing area with black.  IMG is in BGR format.
    #
    rows, cols, _ = img.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = 10*np.random.uniform()-10/2
    #print("Image shape is ", rows, "x", cols, " shifting ", tr_x, " in x ", tr_y, " in y")
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img_tr = cv2.warpAffine(img,Trans_M,(cols,rows))
    label_tr = cv2.warpAffine(label,Trans_M,(cols,rows))
    return img_tr, label_tr

def augment(img_raw, label):
    #
    # We "augment" images during training.
    #
    # imageList is a list of camera image pathnames [left, center, right]
    # y is the label for these three images
    #
    # We choose one of the camera images, augment it to simulate
    # potential environments, then return a chosen image and label y.
    #
    img = augment_brightness_camera_images(img_raw)
    if (np.random.randint(4) == 0):
        img_shadows = add_random_shadow(img)
    else:
        img_shadows = img
    if (np.random.randint(2) == 0):
        img_trans, label_trans = trans_image(img_shadows, label, 25)
    else:
        img_trans, label_trans = img_shadows, label
    if (np.random.randint(4) == 0):
        img_flip = cv2.flip(img_trans, 1)
        label_flip = cv2.flip(label_trans, 1)
    else:
        img_flip, label_flip = img_trans, label_trans
    img_rgb = bgr2rgb(img_flip)
    label_rgb = bgr2rgb(label_flip)
    return img_rgb, label_rgb
