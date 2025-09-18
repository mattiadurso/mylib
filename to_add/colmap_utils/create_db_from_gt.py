import sys
import os
sys.path.append(os.path.expanduser("~")+"/Desktop/Repos/colmap") # path to COLMAP repository
import glob
import numpy as np

import torch
from tqdm.auto import tqdm
from itertools import combinations
from pathlib import Path

from mylib.colmap_utils.eval_colmap import *
from mylib.colmap_utils.database_from_colmap import COLMAPDatabase
from mylib.projections import *
from mylib.datasets.read_datasets import COLMAP_Scene
from mylib.plot import *
from mylib.conversions import to_torch, quaternion_from_matrix_colmap, rotmat_to_axis_angles, axis_angles_to_rotmat



def inject_poses_to_db(dataset_path, run_name, db_name, new_db=True):
    """"
    dataset_path: path to the dataset eg. /HDD/Datasets/Sony2
    run_name: name of the run eg. "run1" eg. /Castle
    db_name: name of the db eg. "database.db" 
    """
    if new_db:
        # copy db before inject
        new_db_name = db_name.split(".db")[0]+"_inj.db"
        os.system(f"cp {dataset_path}/{run_name}/{db_name} {dataset_path}/{run_name}/{new_db_name}")
        db_name = new_db_name 

    
    # loading db
    db_path = f"{dataset_path}/{run_name}/{db_name}"
    db = COLMAPDatabase.connect(db_path)
    cursor = db.cursor()

    # Get a mapping between image ids and image names in db
    image_id_to_name_db = dict()
    cursor.execute("SELECT image_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        name = row[1]
        image_id_to_name_db[image_id] = name
    name_to_image_id_db = {v: k for k,v in image_id_to_name_db.items()}


    # read GT model
    cameras, images, points3D = read_colmap_model(Path(dataset_path, "sparse"), ext='.txt')
    images_indexed_by_name_gt = {images[i].name: images[i] for i in range(1,len(images)+1)}

    scene = dataset_path.split("/")[-1]
    n_images = len(glob.glob(f"{dataset_path}/images/*.jpeg"))

    for i,j in tqdm(combinations(range(1,n_images+1), 2)):
        query_dict = db.two_view_geometry_query2dict(i,j)
        
        # needed since there are more rows than edges in the viewgraph
        if query_dict is None:
            continue

        # get name of image_id 1 and 2 from db
        name1 = image_id_to_name_db[query_dict["image_id1"]]
        name2 = image_id_to_name_db[query_dict["image_id2"]]

        # get global q_vecs
        R1 = images_indexed_by_name_gt[name1].qvec2rotmat() # try to change this maybe?
        R2 = images_indexed_by_name_gt[name2].qvec2rotmat()

        t1 = images_indexed_by_name_gt[name1].tvec
        t2 = images_indexed_by_name_gt[name2].tvec

        # add noise
        # ...

        # compute relative pose
        R = R2@R1.T
        lambda_t = R@(t2-t1)
        t = lambda_t / np.linalg.norm(lambda_t)
       
        # update db
        db.update_two_view_geometry(i,j,
                                    q_vec=quaternion_from_matrix(R), 
                                    t_vec=t,
                                    #E = E,
                                    )
    db.close()


def create_db_from_model(DATASET_PATH, scene, run_name='colmap', 
                         img_folder="images", db_name="database_vanilla", 
                         q_noise_std=0, p_noise_std=0, n_matches=-1):
    """
    Creates a database from a model (cameras, images, points3D).
    args:
    - DATASET_PATH: path to the dataset eg. /HDD/Datasets/Sony2/Castle
    - run_name: name of the run eg. "run1" eg. /db_from_gt
    - scene: instance of COLMAP_Scene
    - img_folder: name of the folder containing the images eg. "images"
    - db_name: name of the db eg. "database"
    - q_noise_std: std of the noise to add to the relative pose
    - p_noise_std: std of the noise to add to the keypoints
    - n_matches: number of points to uniformly sample from the model. if -1 then all the points are used.

    """
    print(DATASET_PATH)

    # read poses from GT
    scene = COLMAP_Scene(DATASET_PATH, gt_folder="sparse", img_folder=img_folder)
    fx, fy, cx, cy = scene.cameras[0].params # there is only one camera in the GT model
    camera = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    camera = scene.K

    # creates new db
    os.makedirs(f"{DATASET_PATH}/{run_name}", exist_ok=True)
    db = COLMAPDatabase.connect(f"{DATASET_PATH}/{run_name}/{db_name}.db")
    db.create_tables()

    # adding cameras
    camera_id1 = db.add_camera(scene.cameras[0].model, 
                            scene.cameras[0].width,
                            scene.cameras[0].height,
                            scene.cameras[0].params[1:])

    # adding images & keypoints
    for k in list(scene.images_by_name.keys()):
        img_id = db.add_image(name    = scene.images_by_name[k].name, 
                            camera_id = camera_id1,
                            image_id  = scene.images_by_name[k].id)
        
        # add noise to the keypoints, if points_noise_std = zero then no noise is added
        keypoints = scene.images_by_name[k].xys
        noise = (torch.randn_like(to_torch(keypoints))*p_noise_std)[0].numpy()
        noisy_keypoints = keypoints + noise
        db.add_keypoints(image_id=img_id,keypoints=noisy_keypoints)
        

    # adding matches
    for i, j in tqdm(combinations(list(scene.images_by_name.keys()), 2)):
        matches_3D = np.intersect1d(scene.images_by_name[i].point3D_ids, scene.images_by_name[j].point3D_ids)
        kpt1_, kpt2_ = [], []
        for match_id in matches_3D:
            kpt1_.append(np.where(scene.images_by_name[i].point3D_ids==match_id))
            kpt2_.append(np.where(scene.images_by_name[j].point3D_ids==match_id))
        print(kpt1_, len(kpt1_), kpt1_[0])
        print()

        kpt1_ = np.array(kpt1_).reshape(-1, 1)
        kpt2_ = np.array(kpt2_).reshape(-1, 1)

        # concat
        m = np.concatenate([kpt1_, kpt2_], axis=1)
        # sample some matches randomly. Also m[::len(m)//n_matches] could be an idea. To have the matches uniformly distributed. Anyway, with at least 70 matches GLOMAP seams to be reasobably good.

        if len(m) > n_matches:
            indexes = np.arange(len(m))
            indexes_to_sample = np.random.choice(indexes, size=n_matches, replace=False)
            m = m[indexes_to_sample]
            db.add_matches(image_id1=scene.images_by_name[i].id, image_id2=scene.images_by_name[j].id, matches=m)
        else:
            # if -1 or len(m) <= n_matches then all the matches are added
            db.add_matches(image_id1=scene.images_by_name[i].id, image_id2=scene.images_by_name[j].id, matches=m)

        # adding relative pose
        img1_id = scene.images_by_name[i].id
        img2_id = scene.images_by_name[j].id

        R1 = scene.images[img1_id].qvec2rotmat()
        t1 = scene.images[img1_id].tvec

        R2 = scene.images[img2_id].qvec2rotmat()
        t2 = scene.images[img2_id].tvec

        R = R2 @ R1.T

        # R -> axis angle -> add noise -> back to R
        noise = torch.randn(3)*q_noise_std    
        noisy_axis_angles = rotmat_to_axis_angles(R) + noise
        R = axis_angles_to_rotmat(noisy_axis_angles).squeeze().numpy()

        q = quaternion_from_matrix_colmap(R)
        t = t2 - R @ t1
        t = t / np.linalg.norm(t)

        tx = np.array([[0.,   -t[2],  t[1]], 
                    [ t[2],   0., -t[0]], 
                    [-t[1], t[0],    0.]])


        E_gt = tx @ R
        F_gt = np.linalg.inv(camera).T @ E_gt @ np.linalg.inv(camera)
        H_gt = R
        
        db.add_two_view_geometry(image_id1=img1_id, image_id2=img2_id, 
                                matches=m,
                                qvec=q, tvec=t, 
                                E=E_gt, F=F_gt, H=H_gt,
                                config=2)
        
    db.commit()
    db.close()


def create_db_from_matches(DATASET_PATH, run_name, camera, keypoints, matches, images_folder="images", remove_db=False):
    """
    Creates a database from a keypoints.
    args:
    - DATASET_PATH: path to the dataset eg. /HDD/Datasets/Sony2
    - run_name: name of the run eg. "run1" eg. /Castle
    - camera: dict with camera parameters {model: model, width: width, height: height, params: params} passuming params = [f, cx, cy].
    - features_dict: dictionary with features and descriptors {camera: intrinsic, i:{name:img_name, keypoints: keypoints, descriptors: descriptors}}
    """


    # remove previous db if present
    if remove_db and os.path.exists(f"{DATASET_PATH}/{run_name}/database_from_{run_name.split('_')[-1]}.db"):
        for path in glob.glob(f"{DATASET_PATH}/{run_name}/database_*"):
            os.system(f"rm {path}")
        print("Old database removed.")
    
    print("Creating new database in:", DATASET_PATH)
    images_path = Path(DATASET_PATH, images_folder)
        
    # create new db
    os.makedirs(f"{DATASET_PATH}/{run_name}", exist_ok=True)
    db = COLMAPDatabase.connect(f"{DATASET_PATH}/{run_name}/database_from_{run_name.split('_')[-1]}.db")
    db.create_tables()

    # adding cameras
    camera_id1 = db.add_camera(camera["model"], camera["width"], camera["height"], camera["params"])
    f, cx, cy = camera["params"]
    camera = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])

    # adding images & keypoints & descriptors
    images_paths = sorted(glob.glob(f"{images_path}/*.jpeg")+glob.glob(f"{images_path}/*.JPEG")+glob.glob(f"{images_path}/*.png")+glob.glob(f"{images_path}/*.PNG"))
    print("Images found:", len(images_paths))
    for id in list(keypoints.keys()):
        #print(f"Adding image {images_paths[id-1].split('/')[-1]} with id {id}.")
        img_id = db.add_image(name      = images_paths[id-1].split('/')[-1], 
                              camera_id = camera_id1,
                              image_id  = id)
        
        db.add_keypoints(image_id= id,   keypoints=keypoints[id])
    
    # adding matches
    n_images = len(images_paths)
    print("Adding matches...")
    for i,j in tqdm(combinations(range(1,n_images+1), 2)):
        
        pair_id = image_ids_to_pair_id(i,j)
        db.add_matches(image_id1=i, image_id2=j, matches=matches[pair_id])

        if len(matches[pair_id]) > 50:
            # compute relative pose
            m = matches[pair_id]
            E = cv2.findEssentialMat(  keypoints[i][m[:,0]], keypoints[j][m[:,1]])[0]
            F = cv2.findFundamentalMat(keypoints[i][m[:,0]], keypoints[j][m[:,1]])[0]
            H = cv2.findHomography(    keypoints[i][m[:,0]], keypoints[j][m[:,1]])[0]
            R, t = cv2.recoverPose(E,  keypoints[i][m[:,0]], keypoints[j][m[:,1]])[1:3]

            # adding relative pose
            db.add_two_view_geometry(image_id1=i, image_id2=j, 
                                    matches=matches[pair_id],
                                    qvec=quaternion_from_matrix(R), tvec=t,
                                    E=E, F=F, H=H,
                                    config=2)
            #print(f"Added relative pose between {i} and {j}.")

   
    db.commit()
    db.close()
    print("Database created.", end="\n\n")


def get_keypoints_and_matches_from_model_utils(images_path, model):
    """
    Get keypoints and matches from am instace of GeneralUtils in mylib.model.models.
    Args:
        images_path: path to the images
        model: instance of GeneralUtils
    Returns:
        keypoints: dict of keypoints for each image
        matches: dict of matches for each pair of images

    """
    ### READING IMAGES
    # len images
    n_images = len(glob.glob(f"{images_path}/*.jpeg"))

    # read all images and store them in a list as tensors
    images = []
    for path in sorted(glob.glob(f"{images_path}/*.jpeg")):
        img = cv2.imread(path)
        images.append(torch.from_numpy(img))

    print(f"Found {len(images)} images")

    ### GETTING KEYPOINTS AND MATCHES
    keypoints = {i:torch.empty(0,2) for i in range(1,n_images+1)}
    matches = {}

    # for each pair of image
    print("Computing matches for each pair...")
    for i,j in tqdm(combinations(range(1,n_images+1), 2)):
        # init match indexes
        m0_index, m1_index = torch.empty(0), torch.empty(0)

        # get matches
        m0, m1 = model.get_mkpts([images[i-1], images[j-1]])

        for kpt in m0:
            # add to the keypoints if not already present rounding to 2 decimals
            if not ((kpt[None]*100).int() == (keypoints[i]*100).int()).all(dim=1).any():
                keypoints[i] = torch.cat([keypoints[i], kpt[None]])

            # store the index of the added keypoint use torch where to get the index
            m0_index = torch.cat([m0_index, torch.where(((kpt[None]*100).int() == (keypoints[i]*100).int()).all(dim=1))[0]]).int()
        
        for kpt in m1:
            if not ((kpt[None]*100).int() == (keypoints[i]*100).int()).all(dim=1).any():
                keypoints[j] = torch.cat([keypoints[j], kpt[None]])
            m1_index = torch.cat([m1_index, torch.where(((kpt[None]*100).int() == (keypoints[j]*100).int()).all(dim=1))[0]]).int()
        
        pair_id = image_ids_to_pair_id(i,j)
        matches[pair_id] = torch.cat([m0_index[None], m1_index[None]]).T


    keypoints = {k:v.numpy() for k,v in keypoints.items()}
    matches   = {k:v.numpy() for k,v in matches.items()}

    return keypoints, matches

# add main for create_db_from_model
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a database from a model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--scene", type=str, required=True, help="Path to the scene folder.")
    parser.add_argument("--run_name", type=str, default='colmap', help="Name of the run.")
    parser.add_argument("--img_folder", type=str, default="images", help="Name of the folder containing the images.")
    parser.add_argument("--db_name", type=str, default="database_vanilla", help="Name of the database.")
    parser.add_argument("--q_noise_std", type=float, default=0.0, help="Standard deviation of the noise to add to the relative pose.")
    parser.add_argument("--p_noise_std", type=float, default=0.0, help="Standard deviation of the noise to add to the keypoints.")
    parser.add_argument("--n_matches", type=int, default=-1, help="Number of matches to sample. If -1, all matches are used.")
    args = parser.parse_args()


    # create db from model
    create_db_from_model(DATASET_PATH=args.dataset_path, 
                         run_name=args.run_name,
                         scene=args.scene, 
                         img_folder=args.img_folder,
                         db_name=args.db_name, 
                         q_noise_std=args.q_noise_std, 
                         p_noise_std=args.p_noise_std, 
                         n_matches=args.n_matches)
    
    print("Database created successfully.")

