import os
import imageio
import numpy as np
import cv2

from preprocess import read_calib, read_image, get_depth_map
from build import Scene

# Good for reading images with PFM format.
imageio.plugins.freeimage.download()

# Find all image directories paths.
dir_path = './MiddEval3/trainingH'
paths = [d.path for d in os.scandir(dir_path) if d.is_dir()]

def write_depth_map(img, path, name):
    '''
    Pass in the single-channel depth map, then create the greyscale image with given path and name.
    The provided name need not to include extension name.
    '''
    path = os.path.join(path, name + '.jpg')
    out = img.copy()
    out[out == 0] = np.max(out)
    out = 255. * (out - np.min(out)) / (np.max(out) - np.min(out))
    cv2.imwrite(path, np.stack([out, out, out], axis=-1))
    print('Depth map ' + path + ' exported.')
    return

def reproject(scene, resolution, camera, K):
    '''
    For given camera position in cam0's camera space. We want to get the reproject image based on given intrinsic matrix.
    '''
    # Reproject each pixel to some depth.
    u, v = np.meshgrid(np.arange(0., resolution[1]), np.arange(0., resolution[0]))
    uv = np.stack([u, v], axis=-1)
    # Build extrinsic.
    E = np.eye(4)
    E[:3,3] = camera
    img = scene.find_nearest_neighbors(uv, K, E)
    return img

if __name__ == '__main__':
    for path in paths:
        # Read calib file.
        calib_path = os.path.join(path, 'calib.txt')
        calib = read_calib(calib_path)
        # Performs preprocessing on the image files.
        img_0, img_1 = read_image(path)
        depth_0, depth_1 = get_depth_map(calib, img_0[:,:,-1], 'cam0'), get_depth_map(calib, img_1[:,:,-1], 'cam1')
        write_depth_map(depth_0, path, 'depth0')
        write_depth_map(depth_1, path, 'depth1')
        img_0[:,:,-1], img_1[:,:,-1] = depth_0, depth_1
        print(path + ' is processed.')
        # Build the scene.
        scene = Scene(calib)
        scene.build(img_0, 'cam0')
        scene.build(img_1, 'cam1')
        scene.show()
        scene.build_tree()
        # Reprojection. Interpolation test.
        K, camera = calib['cam0'], np.array([calib['baseline'] / 2000,0,0])
        K = (K + calib['cam1']) / 2
        #K[0,2] -= .5 * (calib['cam1'][0,2] - K[0,2])
        resolution = (img_0.shape[0], img_0.shape[1])
        img = reproject(scene, resolution, camera, K)
        img = img[:,:,::-1]
        cv2.imwrite(os.path.join(path, 'result_interpolation.jpg'), img)
