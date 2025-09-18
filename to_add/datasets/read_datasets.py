import torch
from copy import deepcopy
import re
import json
import os
import cv2
import numpy as np
import kornia
import random
import glob
import matplotlib.pyplot as plt

from pathlib import Path
from ..colmap_utils.eval_colmap import read_colmap_model
from ..geometry import compute_relative_camera_motion


class ScannetDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, offset=0, invert_pose=True):
        self.name = "scannet"
        self.dataset_path = dataset_path
        self.json_path = (
            "PlaneRecTR/datasets/scannetv1_plane/scannetv1_plane_len760_val.json"
        )
        self.scenes_list = sorted(self.find_scenes_id())
        # self.first_fram_id = first_frame_id//4 *4
        self.offset = offset * 4
        self.invert_pose = invert_pose
        self.imgs_shape = (
            self.__getitem__(0)[0][0].shape[1],
            self.__getitem__(0)[0][0].shape[0],
        )

    def __len__(self):
        return len(self.scenes_list)

    def __getitem__(self, idx):
        scene_id = self.scenes_list[idx % len(self.scenes_list)]
        max_id = len(os.listdir(f"{self.dataset_path}/scene{scene_id}/images")) - 1
        frame0_id = (
            random.randint(4, max_id - self.offset) // 4 * 4
        )  # self.first_fram_id
        frame1_id = frame0_id + self.offset

        # loading images
        imgs_paths = [
            f"{self.dataset_path}/scene{scene_id}/images/frame-{str(frame0_id).zfill(6)}.color.jpg",
            f"{self.dataset_path}/scene{scene_id}/images/frame-{str(frame1_id).zfill(6)}.color.jpg",
        ]
        imgs = [torch.tensor(plt.imread(path)).float() for path in imgs_paths]

        # loading poses between the two images and computing R_GT10
        poses_path = [
            f"{self.dataset_path}/scene{scene_id}/poses/frame-{str(frame0_id).zfill(6)}.pose.txt",
            f"{self.dataset_path}/scene{scene_id}/poses/frame-{str(frame1_id).zfill(6)}.pose.txt",
        ]
        Rt0, Rt1 = [np.loadtxt(path) for path in poses_path]
        if self.invert_pose:
            Rt0, Rt1 = self.invert_P_np(Rt0), self.invert_P_np(Rt1)

        Rt0, Rt1 = (
            torch.from_numpy(Rt0)[None].float(),
            torch.from_numpy(Rt1)[None].float(),
        )
        R0, R1 = Rt1[:, :3, :3], Rt1[:, :3, :3]
        t0, t1 = Rt0[:, :3, 3].reshape(-1, 1), Rt1[:, :3, 3].reshape(-1, 1)
        R_GT, t_GT = kornia.geometry.epipolar.relative_camera_motion(R0, t0, R1, t1)
        K = self.get_K(scene_id)[:, :3, :3]

        return imgs, K, K, Rt0, Rt1, R_GT, scene_id, (frame0_id, frame1_id)

    def find_scenes_id(self):
        """Extract the scene id from the json file"""

        with open(self.json_path, "r") as file:
            data = json.load(file)

        pattern = r"\d{4}_\d{2}"
        scenes_list = []
        for i in range(len(data["annotations"])):
            text = data["annotations"][i]["image_id"]
            scenes_list.append(re.findall(pattern, text)[0])
        return scenes_list

    def get_K(self, scene_id):

        k_path = f"{self.dataset_path}/scene{scene_id}/_info.txt"
        # Dictionary to store the key-value pairs
        data = {}
        # Read lines from the file and parse key-value pairs
        with open(k_path, "r") as file:
            for line in file:
                if "=" in line:
                    key, value = line.strip().split(" = ", 1)  # Split only once
                    data[key.strip()] = value.strip()
        K_int = (
            np.array(data["m_calibrationDepthIntrinsic"].split())
            .astype(np.float32)
            .reshape(4, 4)
        )

        return torch.from_numpy(K_int)[None].float()

    def get_depth(self, scene_id, ids):
        """
        Divide by 1000 to get the depth in meters
        """
        frame0_id, frame1_id = ids
        img0 = (
            cv2.imread(
                f"{self.dataset_path}/scene{scene_id}/depths/frame-{str(frame0_id).zfill(6)}.depth.pgm",
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            / 1000.0
        )
        img1 = (
            cv2.imread(
                f"{self.dataset_path}/scene{scene_id}/depths/frame-{str(frame1_id).zfill(6)}.depth.pgm",
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            / 1000.0
        )
        img0 = cv2.resize(img0, self.imgs_shape)
        img1 = cv2.resize(img1, self.imgs_shape)

        img0 = torch.from_numpy(img0)[None].float()
        img1 = torch.from_numpy(img1)[None].float()

        return [img0, img1]

    def invert_P_np(self, P):
        """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
        Args:
            P: input P matrix
                4x4
        Return:
            P_inv: the inverse of the P matrix
                4x4
        Raises:
            None
        """
        R = P[0:3, 0:3]
        t = P[0:3, 3:4]
        P_inv = np.concatenate((R.T, -R.T.dot(t)), axis=1)
        P_inv = np.concatenate((P_inv, np.array([[0.0, 0.0, 0.0, 1.0]])))
        return P_inv


class ScannetppDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, offset=1, invert_pose=True):
        self.name = "scannetpp"
        self.invert_pose = invert_pose
        self.dataset_path = dataset_path
        self.offset = offset
        self.scenes_list = sorted(os.listdir(self.dataset_path))
        self.len = len(self.scenes_list)
        self.imgs_shape = (
            self.__getitem__(0)[0][0].shape[1],
            self.__getitem__(0)[0][0].shape[0],
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scene_id = self.scenes_list[idx % self.len]  # so I can run the idx from 0 to n
        offset = self.offset
        max_id = len(os.listdir(f"{self.dataset_path}/{scene_id}/iphone/rgb")) - 1
        frame0_id = random.randint(1, max_id - offset)  # no frame 0, start from 1
        frame1_id = frame0_id + offset

        K0, K1, Rt0, Rt1, R_GT10 = self.get_Ks_and_RGT(scene_id, frame0_id, frame1_id)
        img0, img1 = self.get_imgs(scene_id, frame0_id, frame1_id)

        return (
            [img0, img1],
            torch.from_numpy(K0)[None].float(),
            torch.from_numpy(K1)[None].float(),
            torch.from_numpy(Rt0)[None].float(),
            torch.from_numpy(Rt1)[None].float(),
            R_GT10,
            scene_id,
            (frame0_id, frame1_id),
        )

    def get_Ks_and_RGT(self, scene_id, frame0_id, frame1_id):
        json_path = f"{self.dataset_path}/{scene_id}/iphone/pose_intrinsic_imu.json"
        with open(json_path) as f:
            data = json.load(f)

        Rt0 = np.array(data[f"frame_{str(frame0_id).zfill(6)}"]["pose"]).astype(
            np.float32
        )
        Rt1 = np.array(data[f"frame_{str(frame1_id).zfill(6)}"]["pose"]).astype(
            np.float32
        )
        if self.invert_pose:
            Rt0, Rt1 = self.invert_P_np(Rt0), self.invert_P_np(Rt1)
        R0, R1 = torch.from_numpy(Rt0[:3, :3]).float().unsqueeze(0), torch.from_numpy(
            Rt1[:3, :3]
        ).float().unsqueeze(0)
        t0, t1 = torch.from_numpy(Rt0[:3, 3]).float().reshape(-1, 1), torch.from_numpy(
            Rt1[:3, 3]
        ).float().reshape(-1, 1)
        R_GT10, t_GT = kornia.geometry.epipolar.relative_camera_motion(R0, t0, R1, t1)

        return (
            np.array(data[f"frame_{str(frame0_id).zfill(6)}"]["intrinsic"]),
            np.array(data[f"frame_{str(frame1_id).zfill(6)}"]["intrinsic"]),
            Rt0,
            Rt1,
            R_GT10,
        )

    def get_imgs(self, scene_id, frame0_id, frame1_id):
        img0 = plt.imread(
            f"{self.dataset_path}/{scene_id}/iphone/rgb/frame_{str(frame0_id).zfill(6)}.jpg"
        )
        img1 = plt.imread(
            f"{self.dataset_path}/{scene_id}/iphone/rgb/frame_{str(frame1_id).zfill(6)}.jpg"
        )

        img0 = torch.from_numpy(deepcopy(img0)).float()
        img1 = torch.from_numpy(deepcopy(img1)).float()

        return [img0, img1]

    def get_depth(self, scene_id, ids):
        frame0_id, frame1_id = ids
        img0 = plt.imread(
            f"{self.dataset_path}/{scene_id}/iphone/depth/frame_{str(frame0_id).zfill(6)}.png"
        )
        img1 = plt.imread(
            f"{self.dataset_path}/{scene_id}/iphone/depth/frame_{str(frame1_id).zfill(6)}.png"
        )

        img0 = cv2.resize(img0, self.imgs_shape)
        img1 = cv2.resize(img1, self.imgs_shape)

        img0 = torch.from_numpy(img0)[None] * 1000.0
        img1 = torch.from_numpy(img1)[None] * 1000.0

        return [img0, img1]

    def invert_P_np(self, P):
        """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
        Args:
            P: input P matrix
                4x4
        Return:
            P_inv: the inverse of the P matrix
                4x4
        Raises:
            None
        """
        R = P[0:3, 0:3]
        t = P[0:3, 3:4]
        P_inv = np.concatenate((R.T, -R.T.dot(t)), axis=1)
        P_inv = np.concatenate((P_inv, np.array([[0.0, 0.0, 0.0, 1.0]])))
        return P_inv


class COLMAP_Scene(torch.utils.data.Dataset):
    """
    Class to access scenes in COLMAP format
    """

    def __init__(self, dataset_path, gt_folder, img_folder="images/0"):
        """
        dataset_path: str, path to the scene folder eg. HDD/Datasets/ETH3D_Stereo/DSLR_undistorted/botanical_garden
        gt_folder: str, name of the folder containing the colmap model
        """
        self.dataset_path = dataset_path
        self.name = dataset_path.split("/")[-1]
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        # load GT
        self.cameras, self.images, self.points3D = read_colmap_model(
            Path(self.dataset_path, self.gt_folder)
        )
        self.images_by_name = {
            self.images[c].name.split("/")[-1]: self.images[c] for c in self.images
        }
        # # load viewgraph
        # self.edges = self.get_viewgraph_edges()
        # load rgb images as numpy
        self.images_rgb = self.get_scene_imgs()
        # loading intrinsics
        self.K = self.get_intrinsics()

    def __len__(self):
        return self.len

    def __getitem__(self, scene):
        pass

    def get_viewgraph_edges(self):
        edges = []
        try:
            path = glob.glob(str(Path(self.dataset_path, "*", "viewgraph*.txt")))[0]
            with open(path, "r") as file:
                # Read each line in the file
                for line in file:
                    # Print each line
                    i, j = line.replace("\n", "").split(" ")
                    edges.append((i.split("/")[-1], j.split("/")[-1]))
            return edges
        except FileNotFoundError:
            print(
                f"Viewgraph file not found in {Path(self.dataset_path, self.gt_folder)}"
            )
            return edges

    def get_scene_imgs(self):
        imgs = {}
        ext = ["jpg", "png", "jpeg", "JPG"]
        imgs_paths = [
            p
            for e in ext
            for p in glob.glob(str(Path(self.dataset_path, self.img_folder, f"*.{e}")))
        ]

        for img_path in imgs_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs[img_path.split("/")[-1]] = img

        self.images_path = Path(self.dataset_path, self.img_folder)
        return imgs

    def get_intrinsics(self):
        # Return list of intrinsics if more than one, else return the intrinsics matrix.
        intrinsiscs = []

        for i in range(len(self.cameras)):
            fx, fy, cx, cy = self.cameras[list(self.cameras.keys())[i]].params
            intrinsiscs.append(np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1]]))
        intrinsiscs = np.array(intrinsiscs)

        if len(intrinsiscs) == 1:
            intrinsiscs = intrinsiscs[0]

        return intrinsiscs

    def get_relative_pose(self, id1, id2):
        """
        Compute the relative pose between two images
        """
        # get the images
        img1 = self.images[id1]
        img2 = self.images[id2]
        # get the poses
        R1 = img1.qvec2rotmat()
        R2 = img2.qvec2rotmat()
        t1 = img1.tvec.reshape(3, 1)
        t2 = img2.tvec.reshape(3, 1)
        # compute the relative pose
        R_rel, t_rel = compute_relative_camera_motion(R1, R2, t1, t2)
        return R_rel[0].numpy(), t_rel[0].numpy()
