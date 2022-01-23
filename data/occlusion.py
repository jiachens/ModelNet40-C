import open3d as o3d
import numpy as np
from util import get_points, set_points, normalize, shuffle_data



def random_pose(severity):
    """generate a random camera pose"""

    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = np.random.uniform(2./3. * np.pi, 5./6. * np.pi)
    angle_y = 0
    angle_z = np.random.uniform(theta-delta,theta+delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 3., 1)  # select the third column, reshape into (3, 1)-vector

    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T,t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix

def lidar_pose(severity):
    """generate a random LiDAR pose"""
    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = 5./8. * np.pi
    angle_y = 0
    angle_z = np.random.uniform(theta-delta,theta+delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 5, 1)  # select the third column, reshape into (3, 1)-vector
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T,t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix, pose



def get_default_camera_extrinsic():
    return np.array([[1,0,0,1],
                    [0,1,0,0],
                    [0,0,1,2],
                    [0,0,0,1]])


def get_default_camera_intrinsic(width=1920, height=1080):
    return {
        "width": width,
        "height": height,
        "fx": 365,
        "fy": 365,
        "cx": width / 2 - 0.5,
        "cy": height / 2 - 0.5
    }


def core_occlusion(mesh, type, camera_extrinsic=None, camera_intrinsic=None, window_width=1080, window_height=720, n_points=None, downsample_ratio=None):
    if camera_extrinsic is None:
        camera_extrinsic = get_default_camera_extrinsic()
    
    if camera_intrinsic is None:
        camera_intrinsic = get_default_camera_intrinsic()

    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = camera_extrinsic
    camera_parameters.intrinsic.set_intrinsics(**camera_intrinsic)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=window_width, height=window_height)
    viewer.add_geometry(mesh)

    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)
    # viewer.run()

    depth = viewer.capture_depth_float_buffer(do_render=True)

    viewer.destroy_window()
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_parameters.intrinsic, extrinsic=camera_parameters.extrinsic)

    if downsample_ratio is not None:
        ratio =  int((1 - downsample_ratio) / downsample_ratio)
        pcd = pcd.uniform_down_sample(ratio)
    elif n_points is not None:
        # print(np.asarray(pcd.points).shape[0])
        ratio =  int(np.asarray(pcd.points).shape[0] / n_points)
        if ratio > 0:
            # if type == 'occlusion':
            set_points(pcd, shuffle_data(np.asarray(pcd.points)))
            pcd = pcd.uniform_down_sample(ratio)
    
    return pcd


def occlusion_1(mesh, type, severity, window_width=1080, window_height=720, n_points=None, downsample_ratio=None):
    points = get_points(mesh)
    points = normalize(points)
    set_points(mesh, points)
    if type == 'occlusion':
        camera_extrinsic = random_pose(severity)
    elif type == 'lidar':
        camera_extrinsic,pose = lidar_pose(severity)
    camera_intrinsic = get_default_camera_intrinsic(window_width, window_height)
    pcd = core_occlusion(mesh, type, camera_extrinsic=camera_extrinsic, camera_intrinsic=camera_intrinsic, window_width=window_width, window_height=window_height, n_points=n_points, downsample_ratio=downsample_ratio)

    points = get_points(pcd)
    if points.shape[0] < n_points:
        index = np.random.choice(points.shape[0], n_points)
        points = points[index]
    # points = normalize(points)
    # points = denomalize(points, scale, offset)
    if type == 'lidar':
        return points[:n_points,:], pose
    else:
        return points[:n_points,:]

