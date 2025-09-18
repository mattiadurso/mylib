import kornia
import cv2
import torch

from mylib.projections import *
from mylib.geometry import *
from mylib.conversions import to_torch



def mnn(desc1, desc2):
    """
    Multi-nearest neighbors matching
    """
    desc1 = to_torch(desc1, b=False)
    desc2 = to_torch(desc2, b=False)
    return kornia.feature.match_mnn(desc1, desc2)[1].detach().cpu()


def epipolar_mnn(kpt1, kpt2, desc1, desc2, camera_dict1, camera_dict2=None, max_dist=10, return_dicts=False):
    """
    (Re)matching using epipolar geometry after first match with MNN. MNN + epipolar constraint
    """
    params = camera_dict1["params"]
    K1 = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])

    if camera_dict2 is None:
        camera_dict2 = camera_dict1
        K2 = K1
    
    # first match using MNN
    matches_mnn = mnn(desc1, desc2)
    mkpt1_mnn = kpt1[matches_mnn[:,0]]
    mkpt2_mnn = kpt2[matches_mnn[:,1]]


    # compute essential matrix from 1->2 and 2->1
    E12 = compute_essential_poselib(mkpt1_mnn, mkpt2_mnn, camera_dict1, camera_dict2)[0]
    F12 = compute_fundamental_from_essential(E12, K1, K2)
    F21 = F12.permute(0,2,1)

    kpt1 = mkpt1_mnn
    kpt2 = mkpt2_mnn
    desc1 = desc1[matches_mnn[:,0]]
    desc2 = desc2[matches_mnn[:,1]]

    # finding points that are close to the epipolar line 1->2
    candidates1 = {}
    for idx,pt in enumerate(kpt1):
        # epipolar line
        line = compute_epipolar_lines_coeff(F12, pt[None])
        # distance of all other points in imge 2 from the line
        d = distance_line_points_parallel(line, kpt2)
        # keep only (index of) points closer than 3
        points_close_to_epi_line = np.where(d<max_dist)[0]
        candidates1[idx] = {"kpt1":pt, "line":line, "kpt2_index":points_close_to_epi_line, "best_match": None}
    

    # finding points that are close to the epipolar line 2->1
    candidates2 = {}
    for idx,pt in enumerate(kpt2):
        line = compute_epipolar_lines_coeff(F21, pt[None])
        d = distance_line_points_parallel(line, kpt1)
        points_close_to_epi_line = np.where(d<max_dist)[0]
        candidates2[idx] = {"kpt2":pt, "line":line, "kpt1_index":points_close_to_epi_line, "best_match": None}


    # computing the best match, aka the one with the highest dot product among the epipolar neighbors
    for idx,_ in enumerate(kpt1):
        if candidates1[idx]["kpt2_index"].shape[0] == 0:
            continue
        candidates1[idx]["best_match"] = candidates1[idx]["kpt2_index"][
            (desc1[idx]@desc2[candidates1[idx]["kpt2_index"]].T).argmax()].item()

    for idx,_ in enumerate(kpt2):
        if candidates2[idx]["kpt1_index"].shape[0] == 0:
            continue
        candidates2[idx]["best_match"] = candidates2[idx]["kpt1_index"][
            (desc2[idx]@desc1[candidates2[idx]["kpt1_index"]].T).argmax()].item()


    # filtering out the matches that are not mutual
    epipolar_matches = []
    for idx,_ in enumerate(kpt1):
        if candidates1[idx]["best_match"]:
            if candidates2[candidates1[idx]["best_match"]]["best_match"] == idx:
                epipolar_matches.append([idx, candidates1[idx]["best_match"]])
            else:
                candidates1[idx]["best_match"] = None

    if return_dicts:
        return torch.tensor(epipolar_matches), candidates1, candidates2
    
    return torch.tensor(matches_mnn)