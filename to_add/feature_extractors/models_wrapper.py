import torch
import cv2
import kornia
import numpy as np
import pycolmap

from lightglue.utils import load_image, rbd
from lightglue import LightGlue, SuperPoint, DISK
from ..matching import mnn


##########################
###### Main Class ########
##########################

class GeneralUtils():
    # the following classes are special cases of this class
    # this class gives error if used alone. get_Fm is not implemented since changes for each 
    # model but needed
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'


    def get_kpts(self, imgs):
        pass


    def get_Fm(self, kpt1, kpt2):
        Fm, _ = cv2.findFundamentalMat(kpt1.cpu().numpy(), kpt2.cpu().numpy(), cv2.USAC_MAGSAC)#, 1.0, 0.999, 100000)
        Fm = torch.from_numpy(Fm).float().to(self.device)
        #Fm = kornia.geometry.epipolar.find_fundamental(kpt1[None], kpt2[None])
        return Fm
    

    def get_Em(self, ktp1, ktp2, K_int):
        ktp1, ktp2 = ktp1.detach().cpu().numpy(), ktp2.detach().cpu().numpy()
        K_int = K_int[:3, :3]#.detach().cpu().squeeze().numpy()
        Em, _ = cv2.findEssentialMat(ktp1, ktp2, K_int, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
        return torch.from_numpy(Em)[None].float().to(self.device)


    def _forward(self, imgs, K_int0, K_int1, R_GT, method="Fm"):
        kpt1, kpt2 = self.get_kpts(imgs)

        if method=="Fm":
            K_int0, K_int1 = K_int0[:, :3, :3], K_int1[:, :3, :3]
            Fm = self.get_Fm(kpt1, kpt2).cpu()
            Em = kornia.geometry.epipolar.essential_from_fundamental(Fm[None], K_int0, K_int1) 
        elif method=="Em":
            Em = self.get_Em(kpt1, kpt2, K_int0)

        # Motion = Ko.geometry.epipolar.motion_from_essential(Em)[0]
        # err = torch.cat([geodesic_distance_in_degrees(R_GT, Motion[0][1][None].cpu()), 
        #                  geodesic_distance_in_degrees(R_GT, Motion[0][2][None].cpu()),
        #                  ]) #idx 0=1, 2=3, hence only 1,2 are needed. They then become index 0,1 in the err tensor
        return 0 # Motion[0][torch.argmin(err)+1].detach().cpu()
        

##########################
###### Implemented #######
##########################
   
class SPLG_utils(GeneralUtils):
    def __init__(self):
        super().__init__()

        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def get_kpts_and_desc_one_img(self, imgs):
        feats0 = self.extractor.extract(imgs.permute(2,0,1).to(self.device)/255.)
        return feats0["keypoints"].detach().cpu(), feats0["descriptors"].detach().cpu()
    
    def get_matches(self, feats0, feats1):
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return m_kpts0, m_kpts1

    def get_mkpts(self, imgs, return_all=False):
        feats0 = self.extractor.extract(imgs[0].permute(2,0,1).to(self.device)/255.)
        feats1 = self.extractor.extract(imgs[1].permute(2,0,1).to(self.device)/255.)
        
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0 = kpts0[matches[..., 0]].detach().cpu()
        m_kpts1 = kpts1[matches[..., 1]].detach().cpu()
    
        if return_all:
            return kpts0.detach().cpu(), kpts1.detach().cpu(), m_kpts0, m_kpts1, feats0["descriptors"].detach().cpu(), feats1["descriptors"].detach().cpu()
        
        return m_kpts0, m_kpts1


class LoFTR_utils(GeneralUtils):
    """Note: From kornia code it's possible to change the code if needed."""
    def __init__(self, pretrained="indoor_new"): 
        # https://github.com/kornia/kornia/blob/main/kornia/feature/loftr/loftr.py
        super().__init__() 

        self.matcher = kornia.feature.LoFTR(pretrained=pretrained).eval().to(self.device)
    
    def get_mkpts(self, imgs, return_confidence=False):
        img0 = imgs[0].permute(2,0,1)[None].to(self.device)/255.
        img1 = imgs[1].permute(2,0,1)[None].to(self.device)/255.

        img0_gray = kornia.color.rgb_to_grayscale(img0.float())
        img1_gray = kornia.color.rgb_to_grayscale(img1.float())

        img0_res = kornia.geometry.resize(img0_gray, (480, 640), antialias=True)
        img1_res = kornia.geometry.resize(img1_gray, (480, 640), antialias=True)

        input_dict = {
                        "image0": img0_res,  
                        "image1": img1_res,
                    }
        
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].detach().cpu()
        mkpts1 = correspondences["keypoints1"].detach().cpu()

        # rescale matches to original size
        scale_x0 = imgs[0].shape[1] / 640
        scale_y0 = imgs[0].shape[0] / 480
        scale0 = torch.tensor([scale_x0, scale_y0])
        m0_scale = mkpts0 * scale0
        
        scale_x1 = imgs[1].shape[1] / 640
        scale_y1 = imgs[1].shape[0] / 480
        scale1 = torch.tensor([scale_x1, scale_y1])
        m1_scale = mkpts1 * scale1

        if return_confidence:
            return m0_scale, m1_scale, correspondences["confidence"].detach().cpu()
        return m0_scale, m1_scale



##########################
########## TODO ##########
##########################

class SIFT_utils(GeneralUtils):
    def __init__(self): 
        super().__init__() 
        raise NotImplementedError("SIFT not implemented yet.")
    
class RoMa_utils(GeneralUtils):
    def __init__(self): 
        super().__init__() 
        raise NotImplementedError("RoMa not implemented yet.")

class DoDeDo_utils(GeneralUtils):
    def __init__(self): 
        super().__init__() 
        raise NotImplementedError("DoDeDo not implemented yet.")

class ALIKED_utils(GeneralUtils):
    def __init__(self): 
        super().__init__() 
        raise NotImplementedError("ALIKED not implemented yet.")




