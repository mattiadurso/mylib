import os
import time


def run_colmap(
    database_path,
    output_path=None,
    images_path="images",
    mapper="colmap",
    intrisics_path=None,
    feature_extraction=True,
    feature_matching=True,
    matcher="exhaustive",  # 'sequential' or 'exhaustive'
    single_camera_per_folder=True,
    sparse_reconstruction=True,
    dense_reconstruction=False,
    quick_mesh=False,
):
    """
    Run the colmap pipeline on a specific scene from command line.
    Args:
        database_path: path pointing to the project folder in COLMAP format. More at https://colmap.github.io/tutorial.html
        mapper: colmap by default, else lomap or fastmap
        backend: whether to run the backend part of the pipeline
        quick_mesh: whether to use Delaunay (quicker) or Poisson (higher quality) mesher. Works only if backend is True.
        img_folder_name: name of the folder containing the images
        intrisics_path: path to the cameras.txt file. If None, the camera model and parameters will be set to the default values. [currently only one camera per scene is supported]
        no_feature_extraction: whether to skip the feature extraction step. If True, the database.db file must already exist.
        sparse_reconstruction: whether to run the sparse reconstruction step. If False, only the feature extraction step will be run.
    Output:
        None
    Notes:
        - The project folder must contain a folder "frames" with all the images.
        - The function will create a folder called 'glomap' or 'colmap' in the database_path directory
        - The function will write the time taken by each step of the pipeline to a txt
    TODO:
        - Add the possibility to specify the camera model and parameters
    """
    assert mapper in [
        "colmap",
        "glomap",
        "fastmap",
    ], "Mapper must be either 'colmap', 'glomap' or 'fastmap'."

    if "/" not in images_path:
        images_path = f"{database_path}/{images_path}"

    if output_path is None:
        output_path = database_path

    print(f"Running COLMAP pipeline with {mapper} mapper on {database_path}.")
    # print args
    print(
        f"database_path: {database_path}, mapper: {mapper}, images_path: {images_path}, \n\
          intrisics_path: {intrisics_path}, feature_extraction: {feature_extraction}, \n\
          sparse_reconstruction: {sparse_reconstruction}, dense_reconstruction: {dense_reconstruction}, \n\
          quick_mesh: {quick_mesh}"
    )

    os.system(f"mkdir -p {database_path}/sparse")

    # skip feature extraction and matching if db already exists (assuming previous these were done)
    if os.path.exists(f"{database_path}/database.db"):
        feature_extraction = False  # skip feature extraction if db already exists
        # feature_matching = False  # if not atched, colma does it anyway

    ## FRONTEND
    # Extract features
    if feature_extraction is True:
        stime_feat = time.time()

        if intrisics_path is None:
            print(
                f">>> No GT intrinsics provided. {'Assuming' if single_camera_per_folder else 'Not assuming'} images in the same folder share the camera parameters. <<<"
            )
            os.system(
                f"colmap feature_extractor \
                    --database_path {database_path}/database.db \
                    --image_path {images_path} \
                    --SiftExtraction.max_num_features 8192 \
                    --SiftExtraction.max_image_size 4000 \
                    --ImageReader.single_camera_per_folder {int(single_camera_per_folder)} \
                    "
            )

        else:  # with given GT intrisics
            cameras_dict = read_and_parse_cameras_txt(
                intrisics_path
            )  # dict with camera model and params
            if len(cameras_dict) > 1:
                # need to copy images
                print(f">>> Found {len(cameras_dict)} cameras detected. <<<")
                os.rename(
                    f"{images_path}",
                    f"{images_path}_temp",
                )
                os.makedirs(f"{images_path}", exist_ok=True)
                for cam in sorted(cameras_dict.keys()):  # 0,1,2:
                    print(f">>> Copying images for camera {cam} <<<")
                    os.makedirs(f"{images_path}/{cam}", exist_ok=True)
                    os.system(f"cp -r {images_path}_temp/{cam}/* {images_path}/{cam}/")
                    os.system(
                        f"colmap feature_extractor \
                        --database_path {database_path}/database.db \
                        --image_path {images_path} \
                        --SiftExtraction.max_num_features 8192 \
                        --SiftExtraction.max_image_size 4000 \
                        --ImageReader.single_camera_per_folder 1 \
                        --ImageReader.camera_model {cameras_dict[cam]['model']} \
                        --ImageReader.camera_params {cameras_dict[cam]['params']} \
                        "
                    )
                os.system(f"rm -rf {images_path}_temp")  # remove the temporary folder

            else:
                os.system(
                    f"colmap feature_extractor \
                        --database_path {database_path}/database.db \
                        --image_path {images_path} \
                        --SiftExtraction.max_num_features 8192 \
                        --SiftExtraction.max_image_size 4000 \
                        --ImageReader.single_camera_per_folder 1 \
                        --ImageReader.camera_model {cameras_dict[0]['model']} \
                        --ImageReader.camera_params {cameras_dict[0]['params']} \
                        "
                )

        write_time(database_path, "Feature extraction", time.time() - stime_feat)

    stime_match = time.time()
    if matcher in ["sequential", "seq", "s"] and feature_matching:
        # sequential matcher
        os.system(
            f"colmap sequential_matcher \
                    --database_path {database_path}/database.db \
                    --SequentialMatching.overlap 15 \
                    --SequentialMatching.quadratic_overlap 1 \
                    --SequentialMatching.loop_detection 1 \
                    --SequentialMatching.loop_detection_period 10 \
                    --SequentialMatching.loop_detection_num_images 200 \
                    --SequentialMatching.loop_detection_num_nearest_neighbors 5 \
                    --SequentialMatching.loop_detection_num_checks 512 \
                    --SequentialMatching.loop_detection_max_num_features 5000 \
            "
        )
    elif matcher in ["exhaustive", "exaustive", "e"] and feature_matching:
        # exausting matcher
        os.system(
            f"colmap exhaustive_matcher \
                    --database_path {database_path}/database.db"
        )
    elif matcher in ["gps", "spatial"] and feature_matching:
        # GPS matcher
        os.system(
            f"colmap spatial_matcher \
                --database_path {database_path}/database.db \
                --SpatialMatching.max_distance 100 \
            "
        )
    else:
        print(">>> Skipping feature matching step. <<<")
        pass
    write_time(database_path, "Feature matching", time.time() - stime_match)

    if sparse_reconstruction is True:
        # Mapping | COLMAP or GLOMAP |
        stime_map = time.time()
        if mapper == "glomap":
            if intrisics_path is None:
                os.system(
                    f"glomap mapper \
                            --database_path {database_path}/database.db \
                            --image_path {images_path}  \
                            --output_path {output_path}/sparse/ \
                            "
                )
            else:  # with given GT intrisics do not refine them
                os.system(
                    f"glomap mapper \
                            --database_path {database_path}/database.db \
                            --image_path {images_path}  \
                            --output_path {output_path}/sparse/ \
                            --BundleAdjustment.optimize_intrinsics 0 \
                            "
                )

        elif mapper == "fastmap":
            if intrisics_path is None:
                os.system(
                    f"python /home/mattia/Desktop/Repos/fastmap/run.py \
                        --database {database_path}/database.db \
                        --image_dir {images_path}  \
                        --output_dir {output_path}/"
                )  # sparse added by fastmap
            else:
                print(">>> Assuming pinhole camera model. <<<)")
                os.system(
                    f"python /home/mattia/Desktop/Repos/fastmap/run.py \
                        --database {database_path}/database.db \
                        --image_dir {images_path}  \
                        --output_dir {output_path} \
                        --pinhole \
                        --calibrated \
                        "
                )

        else:  # colmap
            if intrisics_path is None:
                os.system(
                    f"colmap mapper \
                        --database_path {database_path}/database.db \
                        --image_path {images_path}  \
                        --output_path {output_path}/sparse/ \
                        --Mapper.ba_refine_focal_length 1 \
                        --Mapper.ba_refine_extra_params 1 \
                        --Mapper.ba_refine_principal_point 1 \
                        --Mapper.ba_local_num_images 10 \
                        "
                )

            else:  # with given GT intrisics do not refine them
                os.system(
                    f"colmap mapper \
                        --database_path {database_path}/database.db \
                        --image_path {images_path}  \
                        --output_path {output_path}/sparse/ \
                        --Mapper.min_num_matches 30 \
                        --Mapper.ba_local_max_num_iterations 50 \
                        --Mapper.ba_global_max_refinements 10 \
                        --Mapper.ba_refine_focal_length 0 \
                        --Mapper.ba_refine_extra_params 0 \
                        --Mapper.ba_refine_principal_point 0 \
                        "
                )

        write_time(output_path, "Mapping", time.time() - stime_map)

    ## BACKEND
    if dense_reconstruction is True:
        os.system(f"mkdir -p {output_path}/dense")
        # undisdtorter
        stime_undist = time.time()
        os.system(
            f"colmap image_undistorter \
                    --image_path {images_path} \
                    --input_path {output_path}/sparse/0 \
                    --output_path {output_path}/dense \
                    --output_type COLMAP \
                    --max_image_size 3840"
        )
        write_time(output_path, "Undistorter", time.time() - stime_undist)

        # stereo
        stime_stereo = time.time()
        os.system(
            f"colmap patch_match_stereo \
                    --workspace_path {output_path}/dense \
                    --workspace_format COLMAP \
                    --PatchMatchStereo.geom_consistency true"
        )
        write_time(output_path, "Stereo", time.time() - stime_stereo)

        # dense
        stime_dense = time.time()
        os.system(
            f"colmap stereo_fusion \
                    --workspace_path {output_path}/dense \
                    --workspace_format COLMAP \
                    --input_type geometric \
                    --StereoFusion.num_threads 16 \
                    --output_path {output_path}/dense/fused.ply"
        )
        write_time(output_path, "Dense", time.time() - stime_dense)

        # these two should be interchangeable
        # Poisson Mesher:
        #       - better suited for creating smooth, high-quality, and watertight surfaces from dense point clouds with normal data. It’s more computationally intensive but produces higher-quality meshes.
        # Delaunay Mesher:
        #       - a faster, simpler algorithm that generates exact triangulations of a point cloud, often used in applications where speed and mesh quality (in terms of triangulation) are prioritized over smoothness and watertightness.

        if quick_mesh:
            # delaunay
            stime_delaunay = time.time()
            os.system(
                f"colmap delaunay_mesher \
                      --input_path {output_path}/dense \
                      --output_path {output_path}/dense/meshed-delaunay.ply"
            )
            write_time(output_path, "Delaunay", time.time() - stime_delaunay)
        else:
            # poisson
            stime_poisson = time.time()
            os.system(
                f"colmap poisson_mesher \
                      --input_path {output_path}/dense/fused.ply \
                      --output_path {output_path}/dense/meshed-poisson.ply"
            )
            write_time(output_path, "Poisson", time.time() - stime_poisson)


def write_time(database_path, step=None, time=None):
    """
    Write the time taken by each step of the pipeline to a txt file.
    Args:
        database_path: path pointing to the project forlder in COLMAP format. More at https://colmap.github.io/tutorial.html
    Output:
        None
    """
    with open(f"{database_path}/time_taken.txt", "a") as f:
        f.write(f"{step}: {time:.3f}s\n")


def read_and_parse_cameras_txt(path_to_txt):
    """
    Read and parse the cameras.txt file. It is assumed to be in COLMAP format.
    Args:
        path_to_txt: Path to the cameras.txt file.
    Returns:
        camera_params: A list of dictionaries containing the camera parameters.
    """
    # Read the txt file
    with open(path_to_txt, "r") as f:
        lines = f.readlines()

    # Extract the camera parameters from the txt file
    camera_params = {}
    for id, line in enumerate(lines):
        # Skip comments
        if line.startswith("#"):
            continue

        # Split the line into tokens
        tokens = line.split()

        # Extract the camera parameters
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = str([float(x) for x in tokens[4:]])[1:-1].replace(" ", "")

        # Store the camera parameters
        camera_params[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "params": params,
        }

    return camera_params
