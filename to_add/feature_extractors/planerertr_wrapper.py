if True:
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    try:
        # ignore ShapelyDeprecationWarning from fvcore
        from shapely.errors import ShapelyDeprecationWarning
        import warnings
        warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    except:
        pass

    from collections import OrderedDict
    from typing import Any, Dict, List, Set
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import copy
    import itertools
    import logging
    import argparse
    import sys
    import kornia
    from tqdm.auto import tqdm
    import kornia as Ko
    import kornia.feature as KF

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.transform import Rotation as R
    from metric import rotation_angle
    import torch.nn as nn
    import torch.nn.functional as F
    import cv2
    import numpy as np
    import kornia as K
    from LightGlue.lightglue.utils import load_image, rbd
    from LightGlue.lightglue import LightGlue, SuperPoint, DISK


    sys.path.append('/home/mattia/Desktop/Models4R/PlaneRecTR/detectron2/')

    import detectron2.utils.comm as comm
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
    from detectron2.engine import (
        DefaultTrainer,
        default_argument_parser,
        default_setup,
        launch,
    )
    from detectron2.evaluation import (
        DatasetEvaluators,
        verify_results,
    )
    from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
    from detectron2.solver.build import maybe_add_gradient_clipping
    from detectron2.utils.logger import setup_logger

    from PlaneRecTR. PlaneRecTR import (
        PlaneSegEvaluator,
        SingleScannetv1PlaneDatasetMapper,
        SingleNYUv2PlaneDatasetMapper,
        SemanticSegmentorWithTTA,
        add_PlaneRecTR_config,
    )


class Trainer(DefaultTrainer):
        """
        Extension of the Trainer class adapted to MaskFormer.
        """
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            """
            Create evaluator(s) for a given dataset.
            This uses the special metadata "evaluator_type" associated with each
            builtin dataset. For your own dataset, you can simply create an
            evaluator manually in your script and do not have to worry about the
            hacky if-else logic here.
            """
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            evaluator_list = []
            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
            # semantic segmentation
            if evaluator_type in ["scannetv1_plane_seg", "nyuv2_plane_seg"]:
                evaluator_list.append(
                    PlaneSegEvaluator(
                        dataset_name,
                        output_dir=output_folder,
                        num_planes= cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, # !
                        vis= True, 
                        vis_period=cfg.TEST.VIS_PERIOD,
                        eval_period=cfg.TEST.EVAL_PERIOD,
                    )
                )
            if len(evaluator_list) == 0:
                raise NotImplementedError(
                    "no Evaluator for the dataset {} with the type {}".format(
                        dataset_name, evaluator_type
                    )
                )
            elif len(evaluator_list) == 1:
                return evaluator_list[0]
            return DatasetEvaluators(evaluator_list)

        @classmethod
        def build_train_loader(cls, cfg):
            if cfg.INPUT.DATASET_MAPPER_NAME == "scannetv1_plane":
                focal_length = 517.97
                offset_x = 320 
                offset_y = 240
                K_int = [[focal_length, 0, offset_x],
                    [0, focal_length, offset_y],
                    [0, 0, 1]]
                mapper = SingleScannetv1PlaneDatasetMapper(cfg, True, intrinsic = np.array(K_int))
                return build_detection_train_loader(cfg, mapper=mapper)

            elif cfg.INPUT.DATASET_MAPPER_NAME == "nyuv2_plane":
                mapper = SingleNYUv2PlaneDatasetMapper(cfg, True)
                return build_detection_train_loader(cfg, mapper=mapper)

            else:
                mapper = None
                return build_detection_train_loader(cfg, mapper=mapper)

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            if cfg.INPUT.DATASET_MAPPER_NAME == "scannetv1_plane":
                focal_length = 517.97
                offset_x = 320 
                offset_y = 240
                K_int = [[focal_length, 0, offset_x],
                    [0, focal_length, offset_y],
                    [0, 0, 1]]
                mapper = SingleScannetv1PlaneDatasetMapper(cfg, is_train=False, intrinsic = np.array(K_int))
                return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            
            elif cfg.INPUT.DATASET_MAPPER_NAME == "nyuv2_plane":
                mapper = SingleNYUv2PlaneDatasetMapper(cfg, is_train=False)
                return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            
            else:
                mapper = None
                return build_detection_test_loader(cfg, dataset_name, mapper = mapper)

        @classmethod
        def build_lr_scheduler(cls, cfg, optimizer):
            """
            It now calls :func:`detectron2.solver.build_lr_scheduler`.
            Overwrite it if you'd like a different scheduler.
            """
            return build_lr_scheduler(cfg, optimizer)

        @classmethod
        def build_optimizer(cls, cfg, model):
            weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
            weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

            defaults = {}
            defaults["lr"] = cfg.SOLVER.BASE_LR
            defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

            norm_module_types = (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
                # NaiveSyncBatchNorm inherits from BatchNorm2d
                torch.nn.GroupNorm,
                torch.nn.InstanceNorm1d,
                torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d,
                torch.nn.LayerNorm,
                torch.nn.LocalResponseNorm,
            )

            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for module_name, module in model.named_modules():
                for module_param_name, value in module.named_parameters(recurse=False):
                    if not value.requires_grad:
                        continue
                    # Avoid duplicating parameters
                    if value in memo:
                        continue
                    memo.add(value)

                    hyperparams = copy.copy(defaults)
                    if "backbone" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER   #! backbone lr*0.1
                    # if "backbone" not in module_name:
                    #     print(module_name)
                    #     pass
                    if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                    ):
                        #print(module_param_name)
                        hyperparams["weight_decay"] = 0.0
                    if isinstance(module, norm_module_types):
                        hyperparams["weight_decay"] = weight_decay_norm
                    if isinstance(module, torch.nn.Embedding):
                        hyperparams["weight_decay"] = weight_decay_embed
                    params.append({"params": [value], **hyperparams}) # ep [{'params': [], 'lr': 0.0001, 'weight_decay': 0.0/0.05}] * 347

            def maybe_add_full_model_gradient_clipping(optim):
                # detectron2 doesn't have full model gradient clipping now
                clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
                )

                class FullModelGradientClippingOptimizer(optim):
                    def step(self, closure=None):
                        all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                        torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                        super().step(closure=closure)

                return FullModelGradientClippingOptimizer if enable else optim

            optimizer_type = cfg.SOLVER.OPTIMIZER
            if optimizer_type == "SGD":
                optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                    params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
                )
            elif optimizer_type == "ADAMW":
                optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                    params, cfg.SOLVER.BASE_LR
                )
            else:
                raise NotImplementedError(f"no optimizer type {optimizer_type}")
            if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                optimizer = maybe_add_gradient_clipping(cfg, optimizer)
            return optimizer

        @classmethod
        def test_with_TTA(cls, cfg, model):
            logger = logging.getLogger("detectron2.trainer")
            # In the end of training, run an evaluation with TTA.
            logger.info("Running inference with test-time augmentation ...")
            model = SemanticSegmentorWithTTA(cfg, model)
            evaluators = [
                cls.build_evaluator(
                    cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
                )
                for name in cfg.DATASETS.TEST
            ]
            res = cls.test(cfg, model, evaluators)
            res = OrderedDict({k + "_TTA": v for k, v in res.items()})
            return res


 # generic class for wrapping a model

class PlaneRecTR_utils():
    def __init__(self):
        self.weights = {
                        "resnet": "PlaneRecTR/checkpoint/PlaneRecTR_r50_pretrained.pth",
                        "swin"  : "PlaneRecTR/checkpoint/PlaneRecTR_swinb_pretrained.pth",
                        "hrnet" : "PlaneRecTR/checkpoint/PlaneRecTR_hrnet32_pretrained.pth",
                    }
        self.args = self.default_argument_parser().parse_args()
        self.cfg = self.setup(self.args)
        self.model = Trainer.build_model(self.cfg)
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            self.weights["resnet"], resume=self.args.resume)


    def default_argument_parser(self, weights=None, epilog=None):
            """
            Create a parser with some common arguments used by detectron2 users.

            Args:
                epilog (str): epilog passed to ArgumentParser describing the usage.

            Returns:
                argparse.ArgumentParser:
            """
            parser = argparse.ArgumentParser(
                epilog=epilog,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            parser.add_argument("--config-file", 
                                default="PlaneRecTR/configs/PlaneRecTRScanNetV1/PlaneRecTR_R50_bs16_50ep.yaml", 
                                metavar="FILE", help="path to config file")
            parser.add_argument("--resume",
                                action="store_true",
                                help="Whether to attempt to resume from the checkpoint directory. "
                                "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",)
            parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
            parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
            parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
            parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

            # PyTorch still may leave orphan processes in multi-gpu training.
            # Therefore we use a deterministic way to obtain port,
            # so that users are aware of orphan processes by seeing the port occupied.
            port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
            parser.add_argument("--dist-url",
                                default="tcp://127.0.0.1:{}".format(port),
                                help="initialization URL for pytorch distributed backend. See "
                                "https://pytorch.org/docs/stable/distributed.html for details.")
            parser.add_argument("--opts",
                                help="""
                                    Modify config options at the end of the command. For Yacs configs, use
                                    space-separated "PATH.KEY VALUE" pairs.
                                    For python-based LazyConfig, use "path.key=value".
                                    """.strip(),
                                default=["MODEL.WEIGHTS", weights],
                                )
            parser.add_argument("--f")

            parser.add_argument('--ds', type=int, default=1, help='0: ScanNet, 1: ScanNet++, 2: ETH3D, 3: Sony')
            parser.add_argument('--offset', type=int, default=12)
            parser.add_argument('--pairs', type=int, default=100)

            return parser


    def setup(self, args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_PlaneRecTR_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
        return cfg
    

    def _forward(self, imgs):
        """
        Function that prepare the images and run the model on them.
        Args:
            imgs: list of images (tensors) of shape [H,W,C]
        Returns:
            y: list of dictionaries with the model's predictions
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #imgs = [cv2.resize(img.numpy(), (256, 192)) for img in imgs]
        imgs = [F.interpolate(img.permute(2,0,1)[None], size=(192, 256), mode='bilinear', align_corners=False)[0] for img in imgs]
        x = [{"image": img.to(device)} for img in imgs]
        with torch.no_grad():
            y = self.model(x)
        return y
    

    def extract_planes(self, outputs_list, i):
        """
        Given the model's output, extract the semantic segmentation, the depth of the planes, the depth of the segmentation, and the normals.
        Args:
            outputs_list: list of outputs from the model
            i: index of the output to consider
        Returns:
            sem_seg: tensor of shape [H,W,Q] planes binary masks, Q is the number of learnable queries (21)
            planes_depth: tensor of shape [H,W] with the depth of the planes
            seg_depth: tensor of shape [H,W] with the depth of the segmentation
            normals: tensor of shape [Q*,3] with the normals, Q* is the number of planes detected
        """
        sem_seg      = outputs_list[i]["sem_seg"].permute(1,2,0).detach().cpu()
        planes_depth = outputs_list[i]["planes_depth"].detach().cpu()
        seg_depth    = outputs_list[i]["seg_depth"].detach().cpu()
        normals      = outputs_list[i]["valid_params"].detach().cpu()
        return sem_seg, planes_depth, seg_depth, normals


    def pad_normals(self, normals0, normals1, normalize=False):
        """
        Padding and normliazing the normals.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            normalize: bool, whether to normalize the normals
        Returns:
            normals0: tensor of normals from image 0 with norm = 1
            normals1: tensor of normals from image 1 with norm = 1
        """
        max_rows = max(normals0.size(0), normals1.size(0))
        max_cols = max(normals0.size(1), normals1.size(1))
        normals0 = F.pad(normals0, (0, max_cols - normals0.size(1), 0, max_rows - normals0.size(0)), value=0)
        normals1 = F.pad(normals1, (0, max_cols - normals1.size(1), 0, max_rows - normals1.size(0)), value=0)
        if normalize:
            normals0 = F.normalize(normals0, dim=1, p=2)
            normals1 = F.normalize(normals1, dim=1, p=2)
        return normals0, normals1


    def order_normals(self, normals0, normals1, idx=None):
        """
        Ordering normals using the Optima Transport Problem and the cosine similarity.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
        Returns:
            normals0: tensor of normals from image 0
            normals1: ordered tensor of normals from image 1 tpo match the ones from image 0
        """
        if len(normals0) != len(normals1):
           normals0, normals1 = self.pad_normals(normals0, normals1)
        cost_function = torch.nn.CosineSimilarity(dim=0)
        features_maskimage0 = normals0 #compute_mask_image_features(sem_seg0, imgs[0], normals0, model_resnet, l)
        features_maskimage1 = normals1 #compute_mask_image_features(sem_seg1, imgs[1], normals1, model_resnet, l)

        similarity_matrix = torch.zeros((len(normals0), len(normals1)))
        for i, vec1 in enumerate(features_maskimage0):
            for j, vec2 in enumerate(features_maskimage1):
                similarity_matrix[i, j] = 1 - cost_function(vec1, vec2)
        # solving transportation problem
        idx0, idx1 = linear_sum_assignment(similarity_matrix, maximize=False)
        # reorder normal
        normals1 = normals1[idx1.tolist()]
        if idx is not None:
            return normals0, normals1, idx1
        return normals0, normals1

 
    def out2ordered_normals(self, out, idx=None):
        """
        Extracting normals from the model's output and ordering them.
        Args:
            out: model's output
        Returns:
            normals0: tensor of normals from image 0
            normals1: ordered tensor of normals from image 1 tpo match the ones from image 0
        """
        sem_seg0, planes_depth0, seg_depth0, normals0 = self.extract_planes(out, 0)
        sem_seg1, planes_depth1, seg_depth1, normals1 = self.extract_planes(out, 1)

        # normalize normals
        normals0 = F.normalize(normals0, dim=1, p=2)
        normals1 = F.normalize(normals1, dim=1, p=2)
        
        # order arrays
        if idx is not None:
            normals0, normals1, idx = self.order_normals(normals0, normals1, idx)
            return normals0, normals1, idx
        
        normals0, normals1 = self.order_normals(normals0, normals1, idx)
        return normals0, normals1
    

    def find_best_rotation(self, normals0, normals1, samples=3):
        """
        Given two sets of matched normals, find the best combination of samples (num of them) that describe the rotation of all of them.
        The method tries all the possible combinations of subsets of sample normals, and according to 
        the mse error between the backprojected normals and the rotated normals, it selects the best rotation.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            samples: number of normals to consider for the rotation estimation
        Returns:
            best_rotations: normals idx, lowest error, rotation matrix
        """
        best_rotations = []
        
        lowest_err = float("inf")
        temp_r = None
        best_normals_idxs = None
        for normals_idxs in itertools.combinations(range(len(normals0)), samples):
            # compute the rotation
            r10, _ = R.align_vectors([normals0[idx].numpy() for idx in normals_idxs], [normals1[idx].numpy() for idx in normals_idxs])
            # exlude normals used in the model from check
            normals0_ex = torch.cat([normals0[i][None] for i in range(len(normals0)) if i not in normals_idxs], dim=0)
            normals1_ex = torch.cat([normals1[i][None] for i in range(len(normals1)) if i not in normals_idxs], dim=0)
            # backproject the normals
            rotated_normals1_ex = torch.from_numpy(r10.apply(normals1_ex))
            normals_cs = torch.cosine_similarity(normals0_ex, rotated_normals1_ex, dim=1)
            error = (normals_cs > 0.99).sum().item() # cant work, there too many elements for a relyable consensus

            if error <= lowest_err:
                lowest_err = error
                best_normals_idxs = normals_idxs
                temp_r = r10.as_matrix()
        # store the best rotation 
        best_rotations.append([best_normals_idxs, lowest_err, temp_r])

        return best_rotations


    def find_best_rotation_given_GT(self, normals0, normals1, R_GT, samples=3):
        """
        Given two sets of matched normals, find the best combination of samples (num of them) that describe the rotation of all of them.
        The method tries all the possible combinations of subsets of sample normals, and according to 
        the distance between the estimated R and the R_GT, it selects the best rotation.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            R_GT: ground truth rotation matrix
            samples: number of normals to consider for the rotation estimation
        Returns:
            best_rotations: normals idx, lowest error, rotation matrix
        """
        best_rotations = []

        min_err = float("inf")
        temp_r = None
        best_normals_idxs = None
        for normals_idxs in itertools.combinations(range(len(normals0)), samples):
            # compute the rotation
            v0 = normals0[list(normals_idxs)]
            v1 = normals1[list(normals_idxs)]
            w = F.softmax(F.cosine_similarity(v0, v1, dim=1), dim=0)
            R_scipy = torch.from_numpy(R.align_vectors(v0.numpy(), v1.numpy(), weights=w)[0].as_matrix().T)[None].float()

            # compute the geodesic distance
            err = geodesic_distance_in_degrees(R_GT, R_scipy).item()

            if err < min_err:
                min_err = err
                best_normals_idxs = normals_idxs
                temp_r = R_scipy
        # store the best rotation 
        best_rotations.append([best_normals_idxs, min_err, temp_r])

        return temp_r#, best_rotations
   

    def get_rotations_wo_GT(self, normals0, normals1, samples=3):
        """
        Compact function to get the best rotation matrix without GT.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            samples: number of normals to consider for the rotation estimation
        Returns:
            r_s: best rotation matrix
        """
        s = self.find_best_rotation(normals0, normals1, samples)
        best_r_s_id = np.argmin([s[i][1] for i in range(len(s))])
        r_s = s[best_r_s_id][-1]
        return r_s


    def get_rotations_w_GT(self, normals0, normals1, R_GT10, samples=3):
        """
        Compact function to get the best rotation matrix with GT.
        Args:
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            R_GT10: ground truth rotation matrix
            samples: number of normals to consider for the rotation estimation
        Returns:
            rs_gt: best rotation matrix
        """
        s_gt = self.find_best_rotation_given_GT(normals0, normals1, R_GT10, samples)
        best_r_s_gt_id = np.argmin([s_gt[i][1] for i in range(len(s_gt))])
        rs_gt = s_gt[best_r_s_gt_id][-1]

        return rs_gt
    

    def reduce_intrisics(self, K, sf):
        """
        Function to reduce the intrinsics matrix based on the scaling factor, assuming a constant aspect ratio.
        Args:
            K0: intrinsics matrix of image 0
            K1: intrinsics matrix of image 1
            sf: scaling factor
        Returns:
            K0_red: reduced intrinsics matrix of image 0
            K1_red: reduced intrinsics matrix of image 1
        """
        sf_m = torch.tensor([1/sf, 0, 0, 0, 1/sf, 0, 0, 0, 1]).view(1,3,3)
        return sf_m@K


    def reproject_2D_to_2D_batch(self, grid, z, Rt0, Rt1, K0, K1):
        """
        Function da project a batch of 2D points in homogeneus coords from one camera frame to another.
        Args:
            p0: 2D points in homogeneus coords
                n, 2
            z: depth of the points. Check proper size.
                B, H, W
            Rt0: extrinsics matrix of the camera frame 0
                B, 4, 4
            Rt1: extrinsics matrix of the camera frame 1
                B, 4, 4
            K0: intrinsics matrix of the camera frame 0 
                B, 3, 3
            K1: intrinsics matrix of the camera frame 1
                B, 3, 3
        Returns:
            p1: 2D points in homogeneus coords in the camera frame 1
                n, 2
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p0 = torch.hstack((grid.cpu(), torch.ones(grid.shape[0], 1))).view(-1,3,1).to(device)
        z0_x, z0_y = p0[:, 0, :].long().cpu(), p0[:, 1, :].long().cpu() 
        z = z[0, z0_y, z0_x].unsqueeze(-1).to(device)
 

        Rt0_torch, Rt1_torch = Rt0.to(device), Rt1.to(device)
        K0_torch, K1_torch = K0.to(device), K1.to(device)
        
        R0_torch = Rt0_torch[:,:3,:3]
        R1_torch = Rt1_torch[:,:3,:3]
        t0_torch = Rt0_torch[:,:3,3].view(-1,3,1)
        t1_torch = Rt1_torch[:,:3,3].view(-1,3,1)

        # realtive camera motion
        R_rel = torch.bmm(R1_torch, R0_torch.transpose(1,2)) 
        t_rel = t1_torch - torch.bmm(R_rel, t0_torch)

        # project p0
        p1_r = K1_torch @ R_rel @ K0_torch.inverse() @ p0
        p1_t = K1_torch @ t_rel / z
        p1 = (p1_r + p1_t).view(-1, 3)
        p1 = p1/p1[:,2].unsqueeze(-1)
        p1 = p1[:,:2]

        return p1.cpu()


    def project_masks(self, sem_seg0, normals0, w, h, z0, Rt0, Rt1, K0, K1):
        """
        NOTICE: Runs in reasoanble time when images are small, but may be optimized.

        Projecting the masks from one image to another.
        Args:
            sem_seg0: tensor of binary masks from image 0
            normals0: tensor of normals from image 0
            w: width of the masks
            h: height of the masks
            z0: depth of the points
            Rt0: extrinsics matrix of the camera frame 0
            Rt1: extrinsics matrix of the camera frame 1
            K0: intrinsics matrix of the camera frame 0
            K1: intrinsics matrix of the camera frame 1
        Returns:
            projected_masks: list of projected masks
        """
        x = torch.arange(0, w)
        y = torch.arange(0, h)
        yy, xx = torch.meshgrid(y,x, indexing='ij')
        grid = torch.stack((xx, yy), dim=2).view(-1,2).float()

        masks0 = self.prepare_masks(sem_seg0, normals0, w, h).unsqueeze(1)
        proj_grid = self.reproject_2D_to_2D_batch(grid, z0, Rt1, Rt0, K1, K0)
        proj_grid_normalized = 2* proj_grid / torch.tensor([w, h]) - 1. # +1*(grid-p1_em).mean(dim=0)
        proj_grid_normalized = proj_grid_normalized.view(h, w, 2).expand(len(normals0), -1, -1, -1)
        projected_masks0_grid_sample = F.grid_sample(masks0, proj_grid_normalized, align_corners=False, padding_mode='zeros').squeeze()

        return projected_masks0_grid_sample 

    
    def compute_IOU_matrix(self, masks0, masks1, theshold=.4, heatmap=False):
        """
        Computing IOU matrix for the masks.
        Args:
            masks0: list of masks from image 0
            masks1: list of masks from image 1
            theshold: theshold for the IOU
        Returns:
            iou_matrix: matrix of IOU values
        """
        iou_matrix = np.zeros((len(masks0), len(masks1)))
        for i in range(len(masks0)):
            for j in range(len(masks1)):
                iou_matrix[i,j] = (masks0[i].bool()*masks1[j].bool()).sum().item() / (masks0[i].bool() + masks1[j].bool()+1e-10).sum().item()
        iou_matrix[iou_matrix < theshold] = 0
        iou_matrix = iou_matrix.T
        if heatmap:
            self.find_matches(iou_matrix, verbose=True)
            sns.heatmap(iou_matrix, annot=True, fmt=".2f")
            plt.xlabel("Masks 0")
            plt.ylabel("Masks 1")
            plt.gca().invert_yaxis()
            plt.show()
            return iou_matrix
        else:
            return iou_matrix
    

    def find_matches(self, IOU_matrix, verbose=False):
        """
        Find the matches between the masks, argmax(row) == argmax(col).
        Args:
            IOU_matrix: matrix of IOU values
            verbose: bool, whether to print the matches
        Returns:
            new_idx0: list of indexes from image 0
            new_idx1: list of indexes from image 1
        """
        new_idx0, new_idx1 = [], []
        if verbose:
            print(f"Img:   0 -> 1", "-------------", sep="\n")
        for i in range(len(IOU_matrix)):
            max_row_id = IOU_matrix[i, :].argmax(-1)
            max_col_id = IOU_matrix[:, max_row_id].argmax(-1)

            if IOU_matrix[i, max_row_id] == IOU_matrix[max_col_id, max_row_id]:
                new_idx0.append(max_row_id)
                new_idx1.append(i)
                if verbose:
                   
                    print(f"Match: {max_row_id} -> {i}")

        return new_idx0, new_idx1


    def prepare_masks(self, sem_seg, normals, w_mask, h_mask, threshold=.51):
        """
        Prepare the masks for the matching setting to zero the values below the threshold.
        Args:
            sem_seg1: tensor of binary masks from image 1
            normals1: tensor of normals from image 1
            w_mask: width of the masks
            h_mask: height of the masks
        Returns:
            masks1: list of masks
        """
        masks = sem_seg.permute(2,0,1)[None].view(-1,1,h_mask,w_mask)[:len(normals)].squeeze()
        masks[masks < threshold] = 0
        #masks1 = F.interpolate(sem_seg1_batch, size=(h, w), mode='bilinear', align_corners=False)[:len(normals1)].squeeze()
        masks[masks < threshold] = 0

        return masks


    def order_normals_with_K_and_Z(self, K0, K1, scaling_factor, sem_seg0, sem_seg1, normals0, normals1, w_mask, h_mask, z0, Rt0, Rt1, verbose=False):
        """
        Ordering normals using the intrinsics and depth.
        Args:
            K0: intrinsics matrix of image 0
            K1: intrinsics matrix of image 1
            scaling_factor: scaling factor
            sem_seg0: tensor of binary masks from image 0
            sem_seg1: tensor of binary masks from image 1
            normals0: tensor of normals from image 0
            normals1: tensor of normals from image 1
            w_mask: width of the masks
            h_mask: height of the masks
            z0: depth of the points
            Rt0: extrinsics matrix of the camera frame 0
            Rt1: extrinsics matrix of the camera frame 1
            verbose: bool, whether to  the matches
        Returns:
            normals0: tensor of normals from image 0
            normals1: ordered tensor of normals from image 1 tpo match the ones from image 0
            projected_masks0: list of projected masks
        """
        # reduce intrinsics and depth
        K0_red, K1_red = self.reduce_intrisics(K0, scaling_factor), self.reduce_intrisics(K1, scaling_factor)
        z0_red = self.reduce_depth(z0, w_mask, h_mask)

        # project masks
        projected_masks0 = self.project_masks(sem_seg0, normals0, w_mask, h_mask, z0_red, Rt0, Rt1, K0_red, K1_red)
        # compare masks
        masks1 = self.prepare_masks(sem_seg1, normals1, w_mask, h_mask)
        IOU_matrix = self.compute_IOU_matrix(projected_masks0, masks1, theshold=.4)
        # order normals
        
        idx0, idx1 = self.find_matches(IOU_matrix, verbose)
        
        return normals0[idx0], normals1[idx1], projected_masks0, idx0, idx1


    def reduce_depth(self, depth, w_mask, h_mask):
        """
        Reduce the depth of the image.
        Args:
            depth: tensor of depth values
            scaling_factor: scaling factor to multiply the depth size
        Returns:
            depth: reduced tensor of depth values
        """
        return F.interpolate(depth[None], size=(h_mask,w_mask), mode='nearest')[0]
    


    def get_Fm_and_R_with_K_and_Z(self, imgs, Rt0, Rt1, K0, K1, z0, R_GT=None, samples=3):
        """
        Getting the fundamental matrix. NOTICE: the translation is taken from ground truth.
        """
        # getting model's predictions and R
        out = self._forward(imgs)
        sem_seg0, _, _, normals0 = self.extract_planes(out, 0)
        sem_seg1, _, _, normals1 = self.extract_planes(out, 1)

        w_mask, h_mask = sem_seg0.shape[1], sem_seg0.shape[0]
        w_img, h_img = imgs[0].shape[1], imgs[0].shape[0]

        normals0, normals1 = self.pad_normals(normals0, normals1, normalize=True)
        normals0_ord, normals1_ord, _, _, _ = self.order_normals_with_K_and_Z(K0, K1, w_img/w_mask, sem_seg0, sem_seg1, normals0, normals1, w_mask, h_mask, z0, Rt0, Rt1)
        
        if R_GT is None: # use all the normals
            R_rel_model = torch.from_numpy(
                R.align_vectors(
                    normals0_ord.numpy(), 
                    normals1_ord.numpy())[0].as_matrix().T
                )[None].float()
            # R_rel_model = self.kabsch_torch_batched(normals0_ord[None], normals1_ord[None])
        else:
            R_rel_model = self.find_best_rotation_given_GT(normals0_ord, normals1_ord, R_GT, samples=samples) # finds the closest rotation to GT computed using a subset of 3 normals

                
        # compute Em
        t0, t1 = Rt0[:,:3,3].view(-1,1), Rt1[:,:3,3].view(-1,1)
        t_rel = t1 - R_rel_model @ t0
        Tx = torch.tensor([[0,           -t_rel[0][2],  t_rel[0][1]],
                           [t_rel[0][2],            0, -t_rel[0][0]],
                           [-t_rel[0][1], t_rel[0][0],            0]
                         ])[None]        
        Em = Tx @ R_rel_model
        Fm = kornia.geometry.epipolar.fundamental_from_essential(Em.float(), K0, K1)
        return Fm, R_rel_model

