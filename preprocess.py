import os
import numpy as np
import cv2
import imageio

def parse_matrix(line):
    '''
    Input is a line representation of matrix in shape [... ... ...;... ... ...;... ... ...]\n
    Returns a numpy matrix.
    '''
    line = line.strip() # Remove the \n at last and white spaces.
    line = line[1:-1] # Remove the left and right brackets.
    line = line.split(';')
    rows = []
    for str_row in line:
        # Parse each line. (Trim is for removing extra space after semi-colon.)
        str_row = str_row.strip().split(' ')
        rows.append(np.array([float(n) for n in str_row]))
    # Create matrix.
    m = np.stack(rows, axis=0)
    return m

def read_calib(path):
    '''
    Input is a string which is the path of calib.txt.
    Returns a dictionary which contains all read parameters in the calib file.
    Use dict.keys() to see what parameters we do have here.
    '''
    d = dict()
    f = open(path, 'r')
    lines = f.readlines() # Read all lines at once.
    for line in lines:
        # Format is 'name=floating#'
        sp = line.split('=')
        name, val = sp[0], sp[1]
        d[name] = parse_matrix(val) if ';' in val else float(val)
    f.close()
    return d

def read_image(path):
    '''
    The inpnt is a string which is the path of the folder which contains all required images.
    The folder must include things called: im0.png, img1.png: which are the images from left
    and right perspective shooting from the baseline camera.
    It also should contain disp0GT.pfm and disp1GT.pfm which are the ground truth disparity map
    of corresponding image.
    The returned image should be in BGRD format where D means disparity instead of depth.
    Color channels are in np.float32 format but within range [0,255];
    WARNING: There might be inf in disparity map.
    '''
    # Read the two images.
    path_0, path_1 = os.path.join(path, 'im0.png'), os.path.join(path, 'im1.png')
    img_0, img_1 = cv2.imread(path_0).astype(np.float32), cv2.imread(path_1).astype(np.float32)
    # Read the disparity map for each image.
    path_0, path_1 = os.path.join(path, 'disp0GT.pfm'), os.path.join(path, 'disp1GT.pfm')
    disp_0, disp_1 = imageio.imread(path_0), imageio.imread(path_1)
    # Concatenate the disparity map onto the image as 4-th channel.
    disp_0, disp_1 = np.expand_dims(disp_0[::-1,:], axis=-1), np.expand_dims(disp_1[::-1,:], axis=-1)
    img_0, img_1 = np.concatenate((img_0, disp_0), axis=-1), np.concatenate((img_1, disp_1), axis=-1)
    return img_0, img_1

def get_depth_map(calib, disparity, camera_label):
    '''
    The inputs are the calib dictionary and the disparity map. The third parameter is the key of intrinsic matrix
    in the calib dictionary.
    We need to use the calib parameters and disparity map to calculate corresponding depth for each pixel.
    The output is still in original shape but with depth instead of disparity.
    '''
    doffs, baseline = calib['doffs'], calib['baseline'] # If calib file is incomplete, then might throw exception.
    disparity += doffs # The length of image line paralleled to baseline.
    m = calib[camera_label]
    assert m.shape == (3,3) # Test matrix shape
    assert m[0,0] == m[1,1] # Test focal length
    f = m[0,0]
    return baseline * f / disparity
