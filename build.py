import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.neighbors import BallTree

from scipy.spatial.distance import cdist

class Scene:
    '''
    A class for convenience to change the scene building process of the project.
    The class should encode all information we need for reprojection.
    '''
    def __init__(self, calib):
        '''
        The input calib dictionary mentioned in main.py/preprocess.py.
        The stored points is currently in XYZ format with shape (#points, 3). And, the color is stored as RGB format.
        '''
        self.__calib = calib
        self.__points = None
        self.__colors = None
        self.__cloud = []
        self.__tree = None
        self.__init_z = None
        self.__far_z = float('inf')
    
    def __build_o3d_cloud(self, img, K, E=None):
        '''
        Build o3d point cloud for visualization. The input image should be in 4-channel BGRD format where D stands for depth.
        The input K is an intrinsic matrix, and E is extrinsic matrix.
        Append the result in __cloud for visualization.
        '''
        # Create intrinsic matrix object.
        camera = o3d.camera.PinholeCameraIntrinsic()
        camera.intrinsic_matrix = K
        # Create o3d object which is convenient to visualize.
        color, depth = img[:,:,:3], img[:,:,3]
        color = o3d.geometry.Image(color[:,:,::-1].astype(np.uint8))
        depth = o3d.geometry.Image(depth.astype(np.float32))
        img = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        if E is None:
            self.__cloud.append(o3d.geometry.PointCloud.create_from_rgbd_image(img, camera))
        else:
            self.__cloud.append(o3d.geometry.PointCloud.create_from_rgbd_image(img, camera, E))
        return
    
    def build_tree(self):
        '''
        Use existed information stored in the object to build a accelerated tree structure.
        '''
        self.__tree = BallTree(self.__points, metric='euclidean')
        self.__init_z = np.min(self.__points, axis=0)[2]
        self.__far_z = np.max(self.__points, axis=0)[2]
        print(self.__far_z)
        print(self.__points)
        print("Ball tree and dictionary built.")
    
    def build(self, img, camera_label):
        '''
        Add all data points to the scene based on calib parameters of given camera_label.
        The input image should be in 4-channel BGRD format where D stands for depth.
        The camera_label should be 'cam0' or 'cam1'.
        '''
        # Directly get intrinsic matrix.
        K = self.__calib[camera_label]
        # Get the extrinsic matrix.
        if camera_label != 'cam0':
            E = np.eye(4)
            E[0,3] = -self.__calib['baseline'] / 1000.
            self.__build_o3d_cloud(img, K, E)
        else:
            self.__build_o3d_cloud(img, K)
        img[:,:,-1] /= 1000 # Same ratio standard for convenience comparing with o3d results.
        u, v = np.meshgrid(np.arange(0., img.shape[1]), np.arange(0., img.shape[0]))
        uvz = np.stack([u, v, img[:,:,-1]], axis=-1)
        uvz = np.reshape(uvz, (uvz.shape[0] * uvz.shape[1], uvz.shape[2]))
        # Extract UV coordinates.
        xyz = uvz.copy()
        xyz[:,2] = 1.
        # Inverse transformation + scale via depth.
        xyz = xyz @ np.linalg.inv(K.transpose()) * np.stack([uvz[:,2], uvz[:,2], uvz[:,2]], axis=-1)
        # Here we have XYZ coordinates in camera space. Now we shift it to global space.
        if camera_label != 'cam0':
            xyz[:,0] += self.__calib['baseline'] / 1000.
        # Get the color information.
        bgr = np.reshape(img[:,:,:3], (img.shape[0] * img.shape[1], 3))
        rgb = bgr[:,::-1]
        # Remove the outliers.
        flag = xyz[:,2] != 0.
        xyz, rgb = xyz[flag], rgb[flag]
        # Save the data.
        if self.__points is None:
            self.__points = xyz
            self.__colors = rgb
        else:
            self.__points = np.concatenate((self.__points, xyz), axis=0)
            self.__colors = np.concatenate((self.__colors, rgb), axis=0)
        # Test the accuracy.
        assert len(self.__colors) == len(self.__points)
        assert np.sum(np.abs(self.__points - np.concatenate([np.asarray(cloud.points) for cloud in self.__cloud], axis=0))) <= 1e-5
        print("Point cloud for " + camera_label + " built.")
        return
    
    def show(self):
        '''
        Use Open3D visualization to test if we are doing great. If the o3d cloud looks right then our cloud also looks fine.
        '''
        o3d.visualization.draw_geometries(self.__cloud)
        return
    
    def find_nearest_neighbors(self, grid, K, E):
        '''
        For given point, we try to find the nearest neighbors.
        '''
        assert self.__tree is not None
        # Initialize 
        z = np.full(grid.shape[:2], self.__init_z).flatten()
        # Get UV coordinates.
        uv = np.reshape(grid, (grid.shape[0] * grid.shape[1], grid.shape[2]))
        # Get XY coordinates.
        xy1 = np.concatenate((uv, np.ones((uv.shape[0], 1))), axis=-1) @ np.linalg.inv(K.transpose())
        E = E.transpose()
        # Get the unit tolerance.
        unit = np.max(np.array([1., 1.]) / np.diag(K)[:2]) * 2
        count, remains = 0, uv.shape[0]
        while remains != 0:
            # Marching.
            while count != 100:
                if count % 10 == 0:
                    print(count)
                # Find corresponding XYZ coordinates of marched ray.
                xyz = (xy1 * np.stack([z, z, z], axis=-1)) @ E[:3] + E[3]
                xyz = xyz[:,:3]
                print(xyz)
                # Get nearest neighbors.
                dist, indices = self.__tree.query(xyz, k=4)
                # Get neighbors.
                targets = self.__points[indices]
                # Get nearest z-step.
                step = targets[:,0,2] - xyz[:,2]
                # Eliminate negative z-step.
                step[step <= 0] = dist[step <= 0,0]
                # Don't march out of the bounds.
                step[z + step > self.__far_z] = self.__far_z - z[z + step > self.__far_z]
                # Don't march the found pixels.
                tol = dist[:,0] < unit * targets[:,0,2]
                step[tol] = 0.
                # Not used in this version (originally used for fill the leaks).
                remains = np.sum(z + step > self.__far_z + 1e-8)
                z += step
                count += 1
            unit *= 2
            z[step > 1e-8] = self.__init_z
            print("Remaining: " + str(remains))
        tol = dist < unit * targets[:,:,2]
        colors = self.__find_nearest_color(indices, tol)
        img = np.reshape(colors, (grid.shape[0], grid.shape[1], 3))
        return img.astype(np.uint8)
    
    def __find_nearest_color(self, indices, tol):
        '''
        Find the correct color and perform interpolation.
        '''
        tol = np.stack([tol, tol, tol], axis=-1)
        colors = self.__colors[indices]
        ret = np.sum(colors * tol, axis=1) / np.sum(tol, axis=1)
        flag = np.sum(tol, axis=1) == 0
        colors = np.squeeze(colors[:,0,:])
        ret[flag] = colors[flag]
        return ret
