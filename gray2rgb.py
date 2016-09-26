# A very simple python script for the colorization of images.
# The function gray2rgb() simply performs colorization followed by joint bilateral upsampling
# in order to match the size of result to that of the original.
# gray2rgb_list() colorizes a list of images.
#
# Prior to use these functions, download the trained .caffemodel from the arthor's website:
# http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
# and save it into ./models.
#
import os, sys
import cv2
import numpy as np

from jbupsample import *

path_pycaffe = '/home/hyunjoon/caffe/python/' # your path to pycaffe
sys.path.append(path_pycaffe)
import caffe

import ipdb

def gray2rgb_list(fn_list_img, gpuid=0, force_gray=False, postfix_orig='', postfix_res='_c'):
    # convert all the images in the list
    #
    with open(fn_list_img, 'r') as fh:
        list_img = fh.readlines()

    n_total_img = len(list_img)

    caffe.set_mode_gpu()
    caffe.set_device(gpuid)

    path_script = os.path.dirname(os.path.abspath(__file__))

    net = caffe.Net(
            os.path.join(path_script, 'models/colorization_deploy_v2.prototxt'), 
            os.path.join(path_script, 'models/colorization_release_v2.caffemodel'), 
            caffe.TEST)

    pts_in_hull = np.load(os.path.join(path_script, 'resources/pts_in_hull.npy')) # load cluster centers
    net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0))

    for (i, img_name) in enumerate(list_img):
        img_name = img_name.strip()
        print '[%d/%d] %s' % (i, n_total_img, img_name)

        r_img = gray2rgb(img_name, gpuid=gpuid, net=net, force_gray=force_gray)
        r_img = np.maximum(0.0, np.minimum(255.0, np.round(r_img * 255.0))).astype('uint8')

        if r_img.size == 0:
            continue

        ttpath, ttname = os.path.split(img_name)
        ttname, ttext = os.path.splitext(ttname)
        orig_name = os.path.join(ttpath, ttname + postfix_orig + ttext)
        res_name = os.path.join(ttpath, ttname + postfix_res + ttext)

        if orig_name != img_name:
            os.rename(img_name, orig_name)
        cv2.imwrite(res_name, r_img)

    
def gray2rgb(fn_img, fn_r_img=None, net=None, force_gray=False, gpuid=0, show_res=False):
    # convert a gray image into RGB
    #
    img = cv2.imread(fn_img)

    hh = img.shape[0]
    ww = img.shape[1]

    if force_gray or _is_gray(img):
        img = np.reshape(img, (hh, ww, -1))
    else:
        return np.zeros(0)

    if img.ndim == 2:
        img = np.tile(img, (1, 1, 3))
    
    o_img = img
    
    if net is None:
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)

        path_script = os.path.dirname(os.path.abspath(__file__))
        print(path_script)

        net = caffe.Net(
                os.path.join(path_script, 'models/colorization_deploy_v2.prototxt'), 
                os.path.join(path_script, 'models/colorization_release_v2.caffemodel'), 
                caffe.TEST)

        pts_in_hull = np.load(os.path.join(path_script, 'resources/pts_in_hull.npy')) # load cluster centers
        net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0))

    (hh_in, ww_in) = net.blobs['data_l'].data.shape[2:] # get input shape
    (hh_out, ww_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

    # prepare input data
    img = cv2.resize(img, (ww_in, hh_in))
    img = img.astype('float32') / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    data_in = img[:, :, 0]
    data_in = np.reshape(data_in - 50.0, (1, 1, hh_in, ww_in))

    # run the colorization, get the result
    net.blobs['data_l'].data[...] = data_in
    net.forward()

    # the size of result is (hh_in, ww_in) = (224, 224)
    ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0))
    img[:, :, 1:3] = cv2.resize(ab_dec, (ww_in, hh_in))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # upsample the result to the size of the input image
    r_img = jbupsample(img, o_img)

    # we use the luminance of the input image, for more natural result
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2LAB)
    o_lab = cv2.cvtColor(o_img.astype('float32') / 255.0, cv2.COLOR_BGR2LAB)
    r_img[:, :, 0] = o_lab[:, :, 0]
    r_img = cv2.cvtColor(r_img, cv2.COLOR_LAB2BGR)

    if fn_r_img is not None:
        cv2.imwrite(fn_r_img, np.maximum(0.0, np.mininum(255.0, r_img*255.0)).astype('uint8'))

    if show_res == True:
        cv2.namedWindow('r_img')
        cv2.imshow('r_img', r_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return r_img


def _is_gray(img):
    if img.ndim == 2:
        return True

    d_bg = abs(img[:, :, 0].astype(float) - img[:, :, 1].astype(float))
    d_br = abs(img[:, :, 0].astype(float) - img[:, :, 2].astype(float))

    if np.mean(d_bg) < 1 and np.mean(d_br) < 1:
        return True

    return False


# if __name__ == '__main__':
    # fn_list_img = '/home/hyunjoon/fd/joint_cascade/data/profile/hyunjoon_list_img.txt'
    # gray2rgb_list(fn_list_img)

    # fn_img = '/home/hyunjoon/fd/joint_cascade/data/profile/img/1000_westernFaces_Collected/g_1000_westernFaces_Collected_0004.jpg'
    # gray2rgb(fn_img, show_res=True)
