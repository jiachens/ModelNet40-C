import os
import numpy as np
import pickle
from config import data_dir, def_ranges
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

example_ids = [6,7,32,54,85]
severity = 4
examples_file = "examples.pkl"


def build_examples(example_ids, severity):
    all_examples = {}
    for example_id in example_ids:
        examples = []
        for corruption in list(def_ranges["corruption"].keys()):
            if corruption == "none":
                continue
            file_path = os.path.join(data_dir, "data_{}_{:d}.npy".format(corruption, severity))
            examples.append(np.load(file_path)[example_id,:])
        all_examples[example_id] = examples

    with open(examples_file, "wb") as f:
        pickle.dump(all_examples, f)
    return examples


def load_examples():
    with open(examples_file, 'rb') as f:
        all_examples = pickle.load(f)
    return all_examples


def rotation_matrix(pitch, yaw, roll):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R


def draw_one_example(example, rotate=[0, 0], scale=1, window_width=1080, window_height=720, show=False, save="test.png", flag=0):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(example[:,:3])

    meshes = []
    for i in range(example.shape[0]):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.0125)
        ball.translate(example[i,:3])
        ball.rotate(rotation_matrix(0, np.pi, np.pi), center=np.array([0,0,0]))
        meshes.append(ball)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height, visible=True)
    for ball in meshes:
        vis.add_geometry(ball)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.90, 0.90, 0.90])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.ZCoordinate

    control = vis.get_view_control()
    # control.convert_from_pinhole_camera_parameters(camera_parameters)
    control.rotate(400, 0)
    control.rotate(0, 100)
    if flag:
        control.rotate(0, -50)
    control.scale(6)
    vis.update_geometry(pcd)
    
    if show:
        vis.run()
    elif save is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save)
        vis.destroy_window()


def draw_one_example_colorful(example, save="test.png"):
    from pointflow_fig_colorful import colorful_pcd
    import mitsuba
    mitsuba.set_variant('scalar_rgb')
    from mitsuba.core import Thread
    from mitsuba.core.xml import load_file
    xml_filename = "tmp.xml"
    colorful_pcd(example, xml_filename)
    Thread.thread().file_resolver().append(os.path.dirname(xml_filename))
    scene = load_file(xml_filename)
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    film = sensor.film()
    from mitsuba.core import Bitmap, Struct
    img = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
    img.write(save)


def draw_examples(tag, examples, colorful=False):
    os.makedirs("figures/{}".format(tag), exist_ok=True)
    for i, example in enumerate(examples):
        if not colorful:
            draw_one_example(example, window_width=720, window_height=600, show=False, save="figures/{}/example_{}.png".format(tag, i))
        else:
            draw_one_example_colorful(example, save="figures/{}/example_{}.png".format(tag, i))

    matplotlib.rcParams.update({'font.size': 13, 'font.weight': 'bold'})
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(15):
        ax = axes[i//5][i%5]
        im = Image.open("figures/{}/example_{}.png".format(tag, i))
        w, h = im.size
        im = im.crop((w * 0.15, h * 0.25, w * 0.85, h * 0.95))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.imshow(im, extent=[0, 1, 0, 1])
        ax.set_title(list(def_ranges["corruption"].values())[i], y=0)
        ax.axis('off')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig("figures/{}/examples.pdf".format(tag))


# Executes once is enough to build the npy file containing examples.
if not os.path.isfile(examples_file):
    build_examples(example_ids)

# Loads examples.
all_examples = load_examples()

# Draws the example demo (one object point cloud with different corruptions).
for example_id in example_ids:
    # Sets `colorful` to True will draw colorful images but require mitsuba module.
    draw_examples(example_id, all_examples[example_id], colorful=False)
