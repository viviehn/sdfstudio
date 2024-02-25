'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import os
import numpy as np
import json
import sys
import tempfile
from pathlib import Path
from argparse import ArgumentParser
import trimesh
import math
import shutil
import imageio
from tqdm import tqdm
from pdb import set_trace as pause

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[3]
sys.path.append(dir_path.__str__())

# from projects.neuralangelo.scripts.convert_data_to_json import _cv_to_gl  # NOQA

from dev.colmap.scripts.python.database import COLMAPDatabase  # NOQA
from dev.colmap.scripts.python.read_write_model import read_model, rotmat2qvec, qvec2rotmat, write_model, write_images_text  # NOQA

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl


class BaseImage:
    def __init__(self, id=None, qvec=None, tvec=None, camera_id=None, name=None, xys=None, point3D_ids=None):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    

def export_to_json(cameras, images, bounding_box, center, radius, file_path):
    intrinsic_param = np.array([camera.params for camera in cameras.values()])
    fl_x = intrinsic_param[0][0]  # TODO: only supports single camera for now
    fl_y = intrinsic_param[0][1]
    cx = intrinsic_param[0][2]
    cy = intrinsic_param[0][3]
    image_width = np.array([camera.width for camera in cameras.values()])
    image_height = np.array([camera.height for camera in cameras.values()])
    w = image_width[0]
    h = image_height[0]
    
    # import pdb; pdb.set_trace()
    
    # k1 = float(intrinsic_param[0][4])
    # k2 = float(intrinsic_param[0][5])
    # k3 = float(intrinsic_param[0][6])
    # k4 = float(intrinsic_param[0][7])
    # "camera_model": "OPENCV_FISHEYE",

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "sk_x": 0.0,  # TODO: check if colmap has skew
        "sk_y": 0.0,
        # "k1": k1, # 0.0,  # take undistorted images only
        # "k2": k2, # 0.0,
        # "k3": k3, # 0.0,
        # "k4": k4, # 0.0,
        "k1": 0.0,  # take undistorted images only
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,  # TODO: not supporting fish eye camera
        "cx": cx,
        "cy": cy,
        "w": int(w),
        "h": int(h),
        "aabb_scale": np.exp2(np.rint(np.log2(radius))),  # power of two, for INGP resolution computation
        "aabb_range": bounding_box,
        "sphere_center": center,
        "sphere_radius": radius,
        "frames": [],
    }

    # read poses
    for img in sorted(images.values()):
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        c2w = _cv_to_gl(c2w)  # convert to GL convention used in iNGP

        frame = {"file_path": "images/" + img.name, "transform_matrix": c2w.tolist()}
        out["frames"].append(frame)

    with open(file_path, "w") as outputfile:
        json.dump(out, outputfile, indent=2)

    return


def create_init_files(db_file, out_dir): # pinhole_dict_file, 
    # Partially adapted from https://github.com/Kai-46//blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # # create template
    # with open(pinhole_dict_file) as fp:
    #     pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} RADIAL {width} {height} {f} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    # for img_name in pinhole_dict:
    #     # w, h, fx, fy, cx, cy, qvec, t
    #     params = pinhole_dict[img_name]
    #     w = params[0]
    #     h = params[1]
    #     fx = params[2]
    #     # fy = params[3]
    #     cx = params[4]
    #     cy = params[5]
    #     qvec = params[6:10]
    #     tvec = params[10:13]

    #     cam_line = cameras_line_template.format(
    #         camera_id="{camera_id}", width=w, height=h, f=fx, cx=cx, cy=cy, k1=0, k2=0)
    #     img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
    #                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
    #                                            image_name=img_name)
    #     template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    # import pdb; pdb.set_trace()
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras, images, points3D = read_model(out_dir, ext='.txt')
    # id2name_dict_old = {i: img.name for i, img in images.items()}

    new_images = {}
    for i, item in images.items():
        # id2name_dict_old[i]
        name = item.name
        id = img_name2id_dict[name]
        # item.id = id
        
        new_images[id] = Image(
            id=id, qvec=item.qvec, tvec=item.tvec,
            camera_id=item.camera_id, name=item.name,
            xys=item.xys, point3D_ids=item.point3D_ids)
        # new_images[id] = Image(**item.__dict__)
        # new_images[id].id = id
        # new_images[id] = item
    
    # new_cameras = {}
    # import pdb; pdb.set_trace()
    
    write_images_text(new_images, os.path.join(out_dir, 'images.txt'))
    
    


    # cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    # images_txt_lines = []
    # for img_name, img_id in img_name2id_dict.items():
    #     image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
    #     images_txt_lines.append(image_line)

    # with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
    #     fp.writelines(cameras_txt_lines)

    # with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
    #     fp.writelines(images_txt_lines)
    #     fp.write('\n')

    # # create an empty points3D.txt
    # fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    # fp.close()


def convert_cam_dict_to_pinhole_dict(cam_dict, pinhole_dict_file):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    print('Writing pinhole_dict to: ', pinhole_dict_file)
    h = 1080
    w = 1920

    pinhole_dict = {}
    for img_name in cam_dict:
        W2C = cam_dict[img_name]

        # params
        fx = 0.6 * w
        fy = 0.6 * w
        cx = w / 2.0
        cy = h / 2.0

        qvec = rotmat2qvec(W2C[:3, :3])
        tvec = W2C[:3, 3]

        params = [w, h, fx, fy, cx, cy,
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def load_COLMAP_poses(cam_file, img_dir, tf='w2c'):
    # load img_dir namges
    names = sorted(os.listdir(img_dir))

    with open(cam_file) as f:
        lines = f.readlines()

    # C2W
    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # header
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
                poses[int(img_idx)]
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5-1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # convert to W2C (follow nerf convention)
        poses_w2c = {}
        for k, v in poses.items():
            poses_w2c[names[k]] = np.linalg.inv(v)
        return poses_w2c


def load_transformation(trans_file):
    with open(trans_file) as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()


def undistort_anon_masks(
    image_dir: Path,
    input_model_dir: Path,
    output_dir: Path,
    colmap_exec: Path = "colmap",
    max_size: int = 2000,
    crop: bool = False,
):
    """Undistort masks using COLMAP.
    args:
        image_dir: Path to the directory containing the masks.
        input_model_dir: Path to the directory containing the COLMAP model.
        output_dir: Path to the directory where the undistorted masks will be saved.
        colmap_exec: Path to the COLMAP executable.
        max_size: Maximum size of the undistorted images.
        crop: Whether to crop the image borders (x-axis) during the undistortion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cameras, images, points3D = read_model(input_model_dir, '.txt')
        new_images = {}
        # Replace the image path (ends with '.JPG') with the mask path (ends with '.png')
        for image_id, image in images.items():
            new_image = image._asdict()
            new_image["name"] = new_image["name"].replace(".JPG", ".png")
            new_images[image_id] = Image(**new_image)

        cur_model_dir = tmpdir / "mask_sparse"
        cur_model_dir.mkdir(parents=True, exist_ok=True)
        write_model(cameras, new_images, points3D, cur_model_dir, ".txt")

        undistort_dir = tmpdir / "mask_undistort"

        # command = (
        #     f"{colmap_exec} image_undistorter"
        #     " --output_type COLMAP"
        #     f" --max_image_size {max_size}"
        #     f" --image_path {image_dir}"
        #     f" --input_path {cur_model_dir}"
        #     f" --output_path {undistort_dir}"
        # )
        # if crop:
        #     command += (
        #         f" --roi_min_x 0.125"
        #         f" --roi_min_y 0"
        #         f" --roi_max_x 0.875"
        #         f" --roi_max_y 1"
        #     )
        # # run_command(command)
        # os.system(command)
        
        
        os.system(f"colmap image_undistorter \
            --image_path {image_dir} \
            --input_path {cur_model_dir} \
            --output_path {undistort_dir} \
            --output_type COLMAP \
            --max_image_size {max_size} \
            --roi_min_x 0.125 \
            --roi_min_y 0 \
            --roi_max_x 0.875 \
            --roi_max_y 1"
                  )

        # # Convert model from .bin to .txt
        # run_command(
        #     f"{colmap_exec} model_converter"
        #     f" --input_path {undistort_dir}/sparse"
        #     f" --output_path {undistort_dir}/sparse"
        #     " --output_type TXT"
        # )

        # Go through all the image masks and make sure they are all 0 or 255
        cameras, images, points3D = read_model(undistort_dir / "sparse")
        for image_id, image in images.items():
            image_path = undistort_dir / "images" / image.name
            mask = imageio.imread(image_path)
            mask = np.array(mask, dtype=np.uint8)
            if (mask == 255).all():
                # The mask is all 255
                continue
            in_between = np.logical_and(mask > 0, mask < 255)
            mask[in_between] = 0
            imageio.imwrite(image_path, mask)

        shutil.move(undistort_dir / "images", output_dir / "masks")


def init_colmap(args):
    assert args.scannetpp, "Provide path to Tanks and Temples dataset"
    scene_list = os.listdir(args.scannetpp)
    scene_list.sort()

    pbar = tqdm(total=len(scene_list))
    for scene in scene_list[1::2]:
    # for scene in ['785e7504b9']:
    # for scene in ['c49a8c6cff']:
        # print(f'Processing scene: {scene}')
        pbar.set_description(desc=f'Scene: {scene}')
        pbar.update(1)
        scene_path = os.path.join(args.scannetpp, scene, 'dslr')
        if os.path.exists(os.path.join(scene_path, 'sdfstudio_transforms.json')):
            continue
        if os.path.exists(os.path.join(scene_path, 'transforms.json')):
            continue

        if not os.path.exists(f"{scene_path}/resized_images"):
            raise Exception(f"'resized_images` folder cannot be found in {scene_path}."
                            "Please check the expected folder structure in DATA_PREPROCESSING.md")

        # # extract features
        # os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
        #         --image_path {scene_path}/resized_images \
        #         --ImageReader.camera_model=OPENCV_FISHEYE \
        #         --SiftExtraction.use_gpu=true \
        #         --SiftExtraction.num_threads=32 \
        #         --ImageReader.single_camera=true"
        #           )
        #         # --ImageReader.camera_model=RADIAL \

        # # match features
        # os.system(f"colmap sequential_matcher \
        #         --database_path {scene_path}/database.db \
        #         --SiftMatching.use_gpu=true"
        #           )

        # # db_file = os.path.join(scene_path, 'database.db')
        # # # sfm_dir = os.path.join(scene_path, 'sparse')
        # # sfm_dir = os.path.join(scene_path, 'colmap')
        # # create_init_files(db_file, sfm_dir) # pinhole_dict_file, 

        # # bundle adjustment
        # os.system(f"colmap point_triangulator \
        #         --database_path {scene_path}/database.db \
        #         --image_path {scene_path}/resized_images \
        #         --input_path {scene_path}/colmap \
        #         --output_path {scene_path}/colmap \
        #         --clear_points 1 \
        #         --Mapper.tri_ignore_two_view_tracks=true"
        #           )
        # os.system(f"colmap bundle_adjuster \
        #         --input_path {scene_path}/colmap \
        #         --output_path {scene_path}/colmap \
        #         --BundleAdjustment.refine_extrinsics=false"
        #           )
        
        if not os.path.exists(f"{scene_path}/masks"):
            # continue
            try:  
                # undistort masks
                undistort_anon_masks(
                    image_dir=Path(f"{scene_path}/resized_anon_masks"),
                    input_model_dir=Path(f"{scene_path}/colmap"),
                    output_dir=Path(f"{scene_path}"),
                    colmap_exec=Path("colmap"),
                    max_size=1440,
                    crop=True,
                )

                # undistortion
                os.system(f"colmap image_undistorter \
                    --image_path {scene_path}/resized_images \
                    --input_path {scene_path}/colmap \
                    --output_path {scene_path} \
                    --output_type COLMAP \
                    --max_image_size 1440 \
                    --roi_min_x 0.125 \
                    --roi_min_y 0 \
                    --roi_max_x 0.875 \
                    --roi_max_y 1"
                        )
            except:
                with open("failed_scenes.txt", "a") as f:
                    f.write(f"{scene}\n")
                continue

        # read for bounding information
        # trans = load_transformation(os.path.join(scene_path, f'{scene}_trans.txt'))
        pts = trimesh.load(os.path.join(scene_path, '../scans/mesh_aligned_0.05.ply'))
        pts = pts.vertices
        # pts_aligned = align_gt_with_cam(pts, trans)
        center, radius, bounding_box = compute_bound(pts[::100])

        # colmap to json
        # cameras, images, points3D = read_model(os.path.join(scene_path, 'colmap'), ext='.txt')
        cameras, images, points3D = read_model(os.path.join(scene_path, 'sparse'), ext='.bin')
        export_to_json(cameras, images, bounding_box, list(center), radius, os.path.join(scene_path, 'transforms.json'))
        # print('Writing data to json file: ', os.path.join(scene_path, 'transforms.json'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scannetpp', type=str, default=None, help='Path to scannet++ dataset')

    args = parser.parse_args()

    init_colmap(args)
