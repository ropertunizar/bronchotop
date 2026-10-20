import glob
import argparse
import cv2
import numpy as np
import torch
from utils_bronchotrack import *
from motrackers import BronchoTracker
from bigtree import find, get_subtree # type: ignore
from collections import OrderedDict

from motrackers.utils.misc import get_centroid
from pathlib import Path
import torch.nn as nn

import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean


from PIL import Image


from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as F
import csv

## Siamese Network used for similarity score computation
class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.cnn1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, 4, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),


                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),


            )

            self.fc1 = nn.Sequential(
                nn.Linear(8*256*256, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 5))

        def forward_once(self, x):
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
            return output

        def forward(self, input1, input2):
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
        
## Function to detect camera motion using RAFT optical flow
def detect_camera_motion(img1, img2, device='cuda'):
    if img1 is None or img2 is None:
        raise ValueError("Failed to load images. Check file paths.")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = F.to_pil_image(img1)
    img2 = F.to_pil_image(img2)

    # Preprocess: use RAFT's weight transforms (scales to [-1, 1] and resizes)
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    img1_t, img2_t = transforms(img1, img2)
    img1_t, img2_t = img1_t.unsqueeze(0).to(device), img2_t.unsqueeze(0).to(device)

    # Load RAFT-small model for speed
    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    # Compute optical flow
    with torch.no_grad():
        flows = model(img1_t, img2_t)
    flow = flows[-1][0].cpu().numpy()  # final flow: shape (2, H, W)

    # Compute radial flow relative to image center
    H, W = flow.shape[1], flow.shape[2]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = W / 2.0, H / 2.0
    vx, vy = flow[0], flow[1]
    dx = xs - cx
    dy = ys - cy
    radial = (vx * dx + vy * dy) / (np.sqrt(dx**2 + dy**2) + 1e-8)
    avg_radial = np.mean(radial)
    direction = 1 if avg_radial > 0 else 0
    return direction

def main(args):
    ## Dataset and image loading, select start and end frames accordingly
    sequence_path = Path(args.sequence_path)
    if not sequence_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {sequence_path}")
    if args.data_type == 'real':
        images = glob.glob(str(sequence_path / "*" / "*.png"))
    else:
        images = glob.glob(str(sequence_path / "frames" / "*" / "*.png"))
    images.sort(key=lambda name: name.split("/")[-1].split("_")[-1])
    end_frame = None if args.end_frame == -1 else args.end_frame
    data_type = args.data_type
    frame_step = 1 if data_type == 'real' else 3
    if args.start_frame < 0 or (end_frame is not None and end_frame <= args.start_frame):
        raise ValueError("Frame range must have a non-negative start and an end greater than the start.")
    images = images[args.start_frame:end_frame:frame_step]
    if not images:
        raise ValueError(f"No PNG frames found in the requested range under {sequence_path}.")

    ## Use full BronchoTop pipeline with Switch Verification Module
    use_sv = args.use_sv

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if data_type == 'real':
        PATH_similarity = args.checkpoint or 'checkpoint_siamese_real/checkpoint_29.pth'
        opt_thresh = 8
        area_thresh = 0.02
    elif data_type == 'sim':
        PATH_similarity = args.checkpoint or 'checkpoint_siamese/checkpoint_29.pth'
        opt_thresh = 20
        area_thresh = 0.01
    else:
        raise ValueError("Invalid data_type. Must be 'real' or 'sim'.")

    # Initialize siamese similarity network
    sim_model = None
    if use_sv:
        checkpoint_path = Path(PATH_similarity)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Siamese checkpoint does not exist: {checkpoint_path}")
        sim_model = SiameseNetwork().to(device)
        checkpoint_sim = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sim_model.load_state_dict(checkpoint_sim, strict=True)
        sim_model.eval()

    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (512, 256))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_labels = []

    # Define airway graph
    alpha = 1e-9
    m = 1
    num_classes = 15
    trans_p = np.loadtxt(args.airway_tree_csv, delimiter=",", dtype=float)
    dist_mat = np.zeros(trans_p.shape)
    for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                    if (trans_p[i][j]<=m):
                            dist_mat[i][j] = (1-alpha*(trans_p[i][j]+1))#/(2*m+1) #*(trans_p[i][j]+1))#
                    else:
                            dist_mat[i][j] = alpha*(trans_p[i][j]+1) #/(num_classes-2*m-1)
    tra = define_airway()
    classes_levels = {0:'TRA',1:'RMB',17:'LMB',2:'TriRUL',6:'BI',26:'LLB6',18:'TriLUL',34:'RLL7',7:'RB4+5',27:'TriLLB',35:'LB1+2',25:'LB4+5',12:'TriRLL',32:'LB9+10',13:'RB9+10',
                          3:'R1', 4:'R2', 5:'R3', 8:'R4', 9:'R5', 10:'R6', 11:'R7', 14:'R8', 15:'R9', 16:'R10', 20:'L6', 19:'L3', 21:'L8', 22:'L1', 23:'L2', 24:'L4', 30:'L5', 28:'L9', 29:'L10'}
    
    classes_dir = {0:0,1:1,2:2,6:3,7:4,12:5,13:6,17:7,18:8,25:9,26:10,27:11,32:12,34:13,35:14,
               3:15,4:16,5:17,8:18,9:19,10:20,11:21,14:22,15:23,16:24,20:25,19:26,21:27,22:28,23:29,24:30,30:31,28:32,29:33}
    classes_name = {}
    for clk in classes_levels:
        clk_name = classes_levels[clk]
        classes_name[clk_name] = classes_dir[clk]
    classes_num = {}
    for cnk in classes_dir:
        classes_num[classes_dir[cnk]] = cnk

    # Initialize tracker, gallery and SV module parameters
    tracker = BronchoTracker()
    frame = 0

    gallery = OrderedDict()

    img_gallery = OrderedDict()
    correct = 0
    loc_scores_track = np.zeros((len(images),15))
    loc_scores_cnn = np.zeros((len(images),15))

    stop_switch = False
    n_stop = 0

    locs = []
    seen_tracks = {}

    direction = []

    prev_back = False
    n_back = 0

    prev_img = []
    
    ## Start processing frames
    for k in range(len(images)):
        
        # Get gt label
        gt = int(Path(images[k]).parent.name)
        gt_loc = classes_levels[gt]
        gt_labels.append(classes_dir[gt])

        # Load image 
        img = cv2.imread(images[k])
        if img is None:
            raise FileNotFoundError(f"Could not read input frame: {images[k]}")
        img = cv2.resize(img,(256,256))
        img_rgb = img.copy()

        # Get lumen detections 
        bboxes, crops, thresholded_w = detect_bboxes(img, area_thresh)
        
        # Tracking
        tracks = tracker.update(bboxes,crops)
        
        if k==0:
            loc = gt_loc
            visible = [child.node_name for child in find(tra, lambda node: node.name == loc).children]
            track_bbox = [t[2][-1] for t in tracks]
            tids = [t[0] for t in tracks]
            visible = visible[::-1]
            if len(tracks)>0:
                if len(tracks) == 1:
                    visible = [loc]
                zipped = list(zip(track_bbox,visible,tids))
            
                res = sorted(zipped, key=lambda x:x[0][0]+x[0][2]/2)
                bbox_sort, lumens, tids_sort = list(zip(*res))
                if len(track_bbox)==1:
                    lumens = [loc]
                gallery[loc] = [[ts,l,bb] for (ts,l,bb) in zip(tids_sort,visible,bbox_sort)]
                loc_scores_track[k] = [1 if ki==int(classes_name[loc]) else 0.0 for ki in range(15)]
            
                probas = [1 if ki==int(classes_name[loc]) else 0.0 for ki in range(15)]

                pre_active = sum([int(t[-1])==0 for t in tracks])
                locs = [loc]
                for idt in tids:
                    seen_tracks[idt] = 5
                forward = True
            else:
                loc_scores_track[k] = [1 if ki==int(classes_name[loc]) else 0.0 for ki in range(15)]
        
        else:
            if k==1:
                    forward = detect_camera_motion(prev_img[0], img, device=str(device))
            else:
                    forward = detect_camera_motion(prev_img[max(k-2,0)], img, device=str(device))

            poped = 0
            new_track = False  
            one_active = False
            for gig in range(len(gallery[loc])):
                tig = gallery[loc][gig-poped][0]
                if not tig in [t[0] for t in tracks]:
                    gallery[loc].pop(gig-poped)
                    poped += 1
            for gig in range(len(gallery[loc])):
                tig = gallery[loc][gig][0]
                tr = tracks[np.where([gallery[loc][ig][0]==tig for ig in range(len(gallery[loc]))])[0][0]]
                if tr[-1]==0:
                    one_active = True
        
            for t in tracks:
                tid = t[0]
                lost = t[-1]
                if not tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                    if lost==0:
                        if tid in seen_tracks:
                            seen_tracks[tid] += 1
                            if (seen_tracks[tid] >= 5 and one_active):
                                new_track = True                                 
                        else:
                            seen_tracks[tid] = 1
            if use_sv and stop_switch:
                new_track = False
       
            loc_root = get_subtree(tra,loc,max_depth=2)
            child_loc = list((loc_root).children)
            vis_loc = [ck.node_name for ck in child_loc]
            vis_loc = vis_loc[::-1]
            track_bbox = []
            tids = []
            tracks_new = []
            for t in tracks:
                if (t[0] in seen_tracks and seen_tracks[t[0]]>=5):
                    track_bbox.append(t[2][-1])
                    tids.append(t[0])
                    tracks_new.append(t)
            tracks = tracks_new
            if len(vis_loc)>0:
                if len(tids)>1:
                    if len(vis_loc)>=len(tids):
                        zipped = list(zip(track_bbox,tids))#[::-1]))
                        res = sorted(zipped, key=lambda x:x[0][0]+x[0][2]/2)
                        bbox_sort, tids_sort = list(zip(*res))
                        lumens = list(vis_loc)
                        bbox_sort = list(bbox_sort)
                        tids_sort = list(tids_sort)
            
                        for it in range(len(tids)):
                            tid = tids_sort[it]
                            if loc in gallery:
                                if tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                                    tig = np.where([gallery[loc][ig][0]==tid for ig in range(len(gallery[loc]))])[0][0]
                                    if new_track :
                                        gallery[loc][tig][1] = lumens[it]
                                    
                                    gallery[loc][tig][2] = bbox_sort[it]
                                else:
                                    gallery[loc].append([tid,lumens[it],bbox_sort[it]])
                            else:
                                    gallery[loc] = [[tid,lumens[it],bbox_sort[it]]]
                    elif len(vis_loc)<len(tids):
                        prev_vis = gallery[loc]
                        prev_bbox = [p[2] for p in prev_vis]
                        prev_lab = [p[1] for p in prev_vis]
                        prev_tid = [p[0] for p in prev_vis]
                        visible_levels = []
                        child_child = []
                        for ck in vis_loc:
                            c_root = get_subtree(tra,ck,max_depth=2)
                            child_c = list((c_root).children)
                            vis_c = [cc.node_name for cc in child_c]
                            for cck in vis_c:
                                child_child.append(cck)
                        for lab in prev_lab:
                            if lab == loc:
                                visible_levels.append(0)
                            elif lab in vis_loc:
                                visible_levels.append(1)
                            elif lab in child_child:
                                visible_levels.append(2)
                        if 0 in visible_levels:
                            bbox_pre = []
                            tids_pre = []
                            lumens = list(vis_loc)
                        
                            for ti in range(len(tids)):
                                if tids[ti] in prev_tid:
                                    tki = np.where(np.array(prev_tid,dtype=int)==tids[ti])[0][0]
                                    lab = prev_lab[tki]
                                    if not (lab==loc):
                                        bbox_pre.append(track_bbox[ti])
                                        tids_pre.append(tids[ti])
                                else:
                                    bbox_pre.append(track_bbox[ti])
                                    tids_pre.append(ti)
                            zipped = list(zip(bbox_pre,tids_pre))#[::-1]))
                            res = sorted(zipped, key=lambda x:x[0][0]+x[0][2]/2)
                            bbox_sort, tids_sort = list(zip(*res))
                            bbox_sort = list(bbox_sort)
                            tids_sort = list(tids_sort)
                            tiloc = np.where(np.array(prev_lab)==loc)[0][0]
                            lumens.append(loc)
                            bbox_sort.append(prev_bbox[tiloc])
                            tids_sort.append(prev_tid[tiloc])
                            for it in range(len(tids_sort)):
                                tid = tids_sort[it]
                                if loc in gallery:
                                    if tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                                        tig = np.where([gallery[loc][ig][0]==tid for ig in range(len(gallery[loc]))])[0][0]
                                        if new_track:
                                            gallery[loc][tig][1] = lumens[it]
                                        
                                        gallery[loc][tig][2] = bbox_sort[it]
                                    else:
                                        gallery[loc].append([tid,lumens[it],bbox_sort[it]])
                                else:
                                        gallery[loc] = [[tid,lumens[it],bbox_sort[it]]]
                        else:
                            new_track_ids = []
                            new_track_bbox = []
                            for tk in range(len(tids)):
                                tc = tids[tk]
                                if not tc in prev_tid:
                                    new_track_ids.append(tc)
                                    new_track_bbox.append(track_bbox[tk])
                            groups = {}
                            for tk in range(len(new_track_ids)):
                                dist_bb = []
                                for bb in prev_bbox:
                                    bb_centroid = get_centroid(bb)
                                    new_centroid = get_centroid(new_track_bbox[tk])
                                    centroid_distances = euclidean(np.array(bb_centroid), np.array(new_centroid))
                                    dist_bb.append(centroid_distances)
                                closest = prev_tid[np.argmin(dist_bb)]
                                if closest in groups:
                                    groups[closest].append(new_track_ids[tk])
                                else:
                                    groups[closest] = [new_track_ids[tk]]
                            updated = np.zeros(len(tids))
                            bbox_sort = []
                            tids_sort = []
                            lumens = []

                            for g in groups:
                                loc_g = prev_lab[np.where(np.array(prev_tid,dtype=int)==int(g))[0][0]]
                                g_root = get_subtree(tra,loc_g,max_depth=2)
                                child_g = list((g_root).children)
                                vis_loc = [ck.node_name for ck in child_g]
                                vis_loc = vis_loc[::-1]
                            
                                bbox_pre = []
                                tids_pre = []
                                for kg in groups[g]:
                                    tids_pre.append(kg)
                                    bbox_pre.append(new_track_bbox[np.where(np.array(new_track_ids,dtype=int)==int(kg))[0][0]])
                                    updated[np.where(np.array(tids,dtype=int)==int(kg))[0][0]] = 1
                                lost_g = tracks[np.where(np.array(prev_tid,dtype=int)==int(g))[0][0]][-1]
                                active = lost_g==0
                                if True: #active:
                                    tids_pre.append(g)
                                    bbox_pre.append(prev_bbox[np.where(np.array(prev_tid,dtype=int)==int(g))[0][0]])
                                    updated[np.where(np.array(tids,dtype=int)==int(g))[0][0]] = 1
                                zipped = list(zip(bbox_pre,tids_pre))#[::-1]))
                                res = sorted(zipped, key=lambda x:x[0][0]+x[0][2]/2)
                                bbox_sub, tids_sub = list(zip(*res))
                                if len(vis_loc)==0:
                                    vis_loc = [loc_g for kloc in tids_sub]
                                for ks in range(len(tids_sub)):
                                    tids_sort.append(tids_sub[ks])
                                    bbox_sort.append(bbox_sub[ks])
                                    lumens.append(vis_loc[ks])
                            for ku in range(len(updated)):
                                if updated[ku]==0:
                                    idu = tids[ku]
                                    bbox = prev_bbox[np.where(np.array(prev_tid,dtype=int)==idu)[0][0]]
                                    lab = prev_lab[np.where(np.array(prev_tid,dtype=int)==idu)[0][0]]
                                    bbox_sort.append(bbox)
                                    tids_sort.append(idu)
                                    lumens.append(lab)
                            while len(lumens)<len(tids_sort):
                                lumens.append(lumens[-1])

                            for it in range(len(tids)):
                                tid = tids_sort[it]
                                if loc in gallery:
                                    if tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                                        tig = np.where([gallery[loc][ig][0]==tid for ig in range(len(gallery[loc]))])[0][0]
                                        if new_track:
                                            gallery[loc][tig][1] = lumens[it]
                                        
                                        gallery[loc][tig][2] = bbox_sort[it]
                                    else:
                                        gallery[loc].append([tid,lumens[it],bbox_sort[it]])
                                else:
                                        gallery[loc] = [[tid,lumens[it],bbox_sort[it]]]
                elif len(tids)==1:
                    if (prev_lum == 1) or (prev_lum>1 and forward):
                        bbox_sort = track_bbox
                        tids_sort = tids
                        if len(gallery[loc])>=1:
                            lumens = [gallery[loc][0][1]]
                        else:
                            lumens = [loc]
                    else:
                        bbox_sort = track_bbox
                        tids_sort = tids
                        lumens = [loc]
                    for it in range(len(tids)):
                                tid = tids_sort[it]
                                if loc in gallery:
                                    if tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                                        tig = np.where([gallery[loc][ig][0]==tid for ig in range(len(gallery[loc]))])[0][0]
                                        gallery[loc][tig][1] = lumens[it]
                                        
                                        gallery[loc][tig][2] = bbox_sort[it]
                                    else:
                                        gallery[loc].append([tid,lumens[it],bbox_sort[it]])
                                else:
                                        gallery[loc] = [[tid,lumens[it],bbox_sort[it]]]

            else:
                lumens_pre = [loc for tk in tids]
                zipped = list(zip(track_bbox,lumens_pre,tids))#[::-1]))
                res = sorted(zipped, key=lambda x:x[0][0]+x[0][2]/2)
                bbox_sort, lumens, tids_sort = list(zip(*res))
                lumens = list(lumens)
                bbox_sort = list(bbox_sort)
                tids_sort = list(tids_sort)
            
                for it in range(len(tids_sort)):
                    tid = tids_sort[it]
                    if loc in gallery:
                        if tid in [gallery[loc][ig][0] for ig in range(len(gallery[loc]))]:
                            tig = np.where([gallery[loc][ig][0]==tid for ig in range(len(gallery[loc]))])[0][0]
                            if new_track:
                                gallery[loc][tig][1] = lumens[it]
                            gallery[loc][tig][2] = track_bbox[it]
                        else:
                            gallery[loc].append([tid,loc,track_bbox[it]])
                    else:
                            gallery[loc] = [[tid,loc,track_bbox[it]]]

            # Likelihood            
            probas = np.array([1/15 for kp in range(15)])   
            cur_lum = len(gallery[loc]) #np.unique([gallery[loc][kg][1] for kg in range(len(gallery[loc]))]))  #len(gallery[loc])

            cur_levels = []
            child_child = []
            vis_loc = np.unique([gallery[loc][kg][1] for kg in range(len(gallery[loc]))])
            labs = [gallery[loc][kg][1] for kg in range(len(gallery[loc]))]
        
            for ck in vis_loc:
                c_root = get_subtree(tra,ck,max_depth=2)
                child_c = list((c_root).children)
                vis_c = [cc.node_name for cc in child_c]
                for cck in vis_c:
                    child_child.append(cck)
            for lab in labs:
                if lab == loc:
                    cur_levels.append(0)
                elif lab in vis_loc:
                    cur_levels.append(1)
                elif lab in child_child:
                    cur_levels.append(2)    

            end_branches = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L8', 'L9', 'L10', 
                            'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10' ]

            if 0 in cur_levels:
                probas[classes_name[loc]] = 1
            else:
                if forward:
                    prev_back = False
                    n_back = 0
                    if prev_lum<=cur_lum:
                        probas[classes_name[loc]] = 1  
                    
                        for klum in range(len(gallery[loc])):

                            lum = classes_name[gallery[loc][klum][1]]
                            lum_name = gallery[loc][klum][1]
                        
                            if not lum_name in end_branches:
                                gid = np.where(np.array([t[0] for t in tracks])==gallery[loc][klum][0])[0][0]
                                tid_lost = tracks[int(gid)][-1]
                            
                                if probas[lum] == 1/15:
                                    probas[lum] = 1-int(tid_lost+1)/6
                                else:
                                    probas[lum] = max(probas[lum],1-int(tid_lost+1)/6)
                    elif prev_lum>cur_lum:

                        if (len(cur_levels)==1) or (len(cur_levels)>1 and np.unique(cur_levels)>=2):
                            probas = loc_scores_track[k-1]
                            probas[classes_name[loc]] = 1/15
                        else:
                            probas[classes_name[loc]] = 1  
                       
                            for klum in range(len(gallery[loc])):
                                lum = classes_name[gallery[loc][klum][1]]
                                lum_name = gallery[loc][klum][1]
                                if not lum_name in end_branches:
                                    gid = np.where(np.array([t[0] for t in tracks])==gallery[loc][klum][0])[0][0]
                                
                                    tid_lost = tracks[int(gid)][-1]
                                
                                    if lum_name in vis_loc:
                                        if probas[lum] == 1/15:
                                            probas[lum] = 1-int(tid_lost+1)/6
                                        else:
                                            probas[lum] = max(probas[lum],1-int(tid_lost+1)/6)
                                    else:
                                        parent = find(tra,lambda node: node.name == loc).parent.name
                                        p_num = classes_name[parent]
                                        if probas[p_num] == 1/15:
                                            probas[p_num] = 1-int(tid_lost+1)/6
                                        else:
                                            probas[p_num] = max(probas[p_num],1-int(tid_lost+1)/6)

                else:
                    if prev_lum==1 and cur_lum>1:
                        prev_back = True
                        n_back += 1
                        probas[classes_name[loc]] = 1
                    else:
                        if prev_back:
                            n_back += 1
                            if n_back >= 10:
                                node_loc = find(tra,lambda node: node.name == loc).parent
                                if not node_loc==None:
                                    probas[classes_name[node_loc.name]] = 1
                                prev_back = False
                                n_back = 0
                            else:
                                probas[classes_name[loc]] = 1
                        else:    
                            probas[classes_name[loc]] = 1
                            prev_back = False
                            n_back = 0         
        
            sum_post_list = []
            num_classes = 15
            states = [si for si in range(num_classes)]
            for s in range(len(states)):
                    dist_t = [dist_mat[s][prev_s]*loc_scores_track[k-1][prev_s] for prev_s in states]
                    sum_post = sum(dist_t)
                    sum_post_list.append(sum_post) 
                    loc_scores_track[k][s] = probas[s]
        
            loc_scores_track[k] = loc_scores_track[k]/loc_scores_track[k].sum()

            # Switch Verification
        
            if use_sv:
                from skimage import metrics
                active_tracklets = sum([int(t[-1])==0 for t in tracks])
            
                pre_active = active_tracklets 
                if classes_levels[classes_num[np.argmax(loc_scores_track[k])]] != loc: #  (prev_lum>1 and cur_lum==1) or (prev_lum==1 and cur_lum>1): # Possible switch
                    last_branching = prev_img[max(0,k-5)]
                    transform_sim = transforms.Compose([transforms.Resize((256,256)),
                                                                    transforms.ToTensor()
                                                                    ])
                    img_pre = transform_sim(Image.fromarray(last_branching).convert('L')).unsqueeze(0).to(device)
                    img_cur = transform_sim(Image.fromarray(img).convert('L')).unsqueeze(0).to(device)
                    if sim_model is None:
                        raise RuntimeError("Switch verification is enabled but its model was not initialized.")
                    output1,output2 = sim_model(img_pre,img_cur)
                    sim_score = torch.nn.functional.pairwise_distance(output1, output2).item()
                
                    if round(sim_score, 2)<=opt_thresh:
                        loc_scores_track[k] = [1/15 for ki in range(15)]
                        loc_scores_track[k][classes_name[loc]] = 1
                        loc_scores_track[k] = loc_scores_track[k]/loc_scores_track[k].sum()
                        stop_switch = True
                        n_stop = 0
                        stop_probas = probas.copy()
                        stop_lum = cur_lum
                        stop_dir = forward
                        stop_sim = sim_score

                if stop_switch:
                    if n_stop>0:
                        if cur_lum == stop_lum:
                            n_stop += 1
                        else:
                            stop_switch = False
                            n_stop = 0
                        if n_stop>11:
                            loc_scores_track[k] = [1/15 for ki in range(15)]
                            loc_scores_track[k] = stop_probas
                            loc_scores_track[k][classes_name[loc]] = 0
                            loc_scores_track[k] = loc_scores_track[k]/loc_scores_track[k].sum()
                            stop_switch = False
                            n_stop = 0
                    else:
                        n_stop += 1

        # Save new location
        loc_prev = loc
        loc = classes_levels[classes_num[np.argmax(loc_scores_track[k])]]       
        gallery[loc] = gallery[loc_prev]

        # Visualization
        final_frame = mask_overlay(img_rgb,thresholded_w) 

        g = gallery[loc]
        color_loc = (0,255,0) if loc==gt_loc else (0,0,255)
        for indt in range(len(g)):
            t = g[indt]
            tid = t[0]
            label = t[1] 
            bbox = t[2]
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            xc = int(x+w/2)
            yc = int(y+h/2)
            cv2.rectangle(final_frame, (x, y), (x + w, y + h), color_loc, 2) 
            cv2.circle(final_frame,(xc,yc),2,color_loc,2)
            cv2.putText(final_frame, str(label),(xc,yc),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_loc,1, cv2.LINE_AA)
        
        ## Plot tree
        tree_img = cv2.imread(args.tree_image)
        if tree_img is None:
            raise FileNotFoundError(f"Could not read airway tree image: {args.tree_image}")
    
        # Legend
        cv2.rectangle(tree_img, (138,188),(256,256),(0,0,0),1)

        cv2.circle(tree_img, center=(150,202), radius=7, thickness = 2, color=(255,0,0))
        cv2.putText(tree_img,'Ground-truth',(162,205),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 0),1, cv2.LINE_AA)

        cv2.rectangle(tree_img, (143,214), (158,225), thickness = 2, color=(0,0,0))
        cv2.putText(tree_img,'Estimated',(162,222),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 0),1, cv2.LINE_AA)

        cv2.line(tree_img, (142,235), (162,235),(0,255,0),4)
        cv2.putText(tree_img,'Correct',(166,237),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 0),1, cv2.LINE_AA)

        cv2.line(tree_img, (142,248), (162,248),(0,0,255),4)
        cv2.putText(tree_img,'Incorrect',(166,250),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 0),1, cv2.LINE_AA)

        # GT
        if str(gt_loc)=='TRA':
            cv2.circle(tree_img,center=(126,50),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='RMB':
            cv2.circle(tree_img,center=(63,75),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='TriRUL':
            cv2.circle(tree_img,center=(37,104),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='BI':
            cv2.circle(tree_img,center=(80,104),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='RLL7':   
            cv2.circle(tree_img,center=(63,139),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='TriRLL':
            cv2.circle(tree_img,center=(63,169),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='RB9+10': 
            cv2.circle(tree_img,center=(63,198),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='RB4+5':
            cv2.circle(tree_img,center=(99,139),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='LMB':
            cv2.circle(tree_img,center=(182,75),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='LLB6':
            cv2.circle(tree_img,center=(156,104),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='TriLUL':
            cv2.circle(tree_img,center=(216,104),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='TriLLB':
            cv2.circle(tree_img,center=(156,134),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='LB1+2':
            cv2.circle(tree_img,center=(200,134),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='LB4+5':
            cv2.circle(tree_img,center=(234,134),radius=12,color=(255,0,0),thickness=2)
        elif str(gt_loc)=='LB9+10':
            cv2.circle(tree_img,center=(156,168),radius=12,color=(255,0,0),thickness=2)

        # Loc
        
        if str(loc)=='TRA':
            cv2.rectangle(tree_img,(111,35),(141,65),color=color_loc,thickness=2)
        elif str(loc)=='RMB':
            cv2.rectangle(tree_img,(48,60),(78,90),color=color_loc,thickness=2)
        elif str(loc)=='TriRUL':
            cv2.rectangle(tree_img,(22,89),(52,119),color=color_loc,thickness=2)
        elif str(loc)=='BI':
            cv2.rectangle(tree_img,(65,89),(95,119),color=color_loc,thickness=2)
        elif str(loc)=='RLL7':   
            cv2.rectangle(tree_img,(48,124),(78,154),color=color_loc,thickness=2)
        elif str(loc)=='TriRLL':
            cv2.rectangle(tree_img,(48,154),(78,184),color=color_loc,thickness=2)
        elif str(loc)=='RB9+10': 
            cv2.rectangle(tree_img,(48,183),(78,213),color=color_loc,thickness=2)
        elif str(loc)=='RB4+5':
            cv2.rectangle(tree_img,(84,124),(114,154),color=color_loc,thickness=2)
        elif str(loc)=='LMB': #(182,75)
            cv2.rectangle(tree_img,(167,60),(197,90),color=color_loc,thickness=2)
        elif str(loc)=='LLB6':
            cv2.rectangle(tree_img,(141,89),(171,119),color=color_loc,thickness=2)
        elif str(loc)=='TriLUL':
            cv2.rectangle(tree_img,(201,89),(231,119),color=color_loc,thickness=2)
        elif str(loc)=='TriLLB':
            cv2.rectangle(tree_img,(141,119),(171,149),color=color_loc,thickness=2)
        elif str(loc)=='LB1+2':
            cv2.rectangle(tree_img,(185,119),(215,149),color=color_loc,thickness=2)
        elif str(loc)=='LB4+5':
            cv2.rectangle(tree_img,(219,119),(249,149),color=color_loc,thickness=2)
        elif str(loc)=='LB9+10':
            cv2.rectangle(tree_img,(141,153),(171,183),color=color_loc,thickness=2)
        
        concat_frame = np.concatenate((final_frame, tree_img), axis=1)
        out.write(concat_frame)

        frame += 1
        if loc==gt_loc:
            correct += 1
        prev_lum = len(gallery[loc]) 
        new_track = False
        if k==0:
            prev_img = [img]
        else:
            if len(prev_img)==0:
                prev_img = [img]
            else: 
                prev_img.append(img)

    # Sequence metrics 
    clas_acc1 = []
    clas_acc3 = []
    clas_acc5 = []
    for t in range(loc_scores_track.shape[0]):
        maxk = 5
        topk = (1,3,5)
        v, pred = torch.from_numpy(loc_scores_track[t]).topk(maxk, -1, True, True)
        pred = pred.t()
    
        correct = pred.eq(gt_labels[t])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum()
            res.append(correct_k * 100.0)
    
        clas_acc1.append(res[0].item())
        clas_acc3.append(res[1].item()) 
        clas_acc5.append(res[2].item())  

        with open(results_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([pred[0].item(), gt_labels[t]])
    out.release()
    print(np.mean(clas_acc1), np.mean(clas_acc3), np.mean(clas_acc5))

def build_parser():
    parser = argparse.ArgumentParser(description="Run the BronchoTop airway-localization pipeline.")
    parser.add_argument(
        "--sequence_path",
        required=True,
        help="Path to the sequence images."
    )
    parser.add_argument("--data_type", choices=("real", "sim"), default="real")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame index to process.")
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Exclusive frame index to stop at; use -1 to process through the last frame.",
    )
    parser.add_argument("--checkpoint", help="Override the default Siamese checkpoint.")
    parser.add_argument("--airway_tree_csv", default="bronchial_tree_simplified.csv")
    parser.add_argument("--tree_image", default="tree_image.png")
    parser.add_argument("--output_video", default="tmp.mp4")
    parser.add_argument("--results_csv", default="bronchotop.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--use-sv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable switch verification.",
    )
    return parser

if __name__ == "__main__":
    main(build_parser().parse_args())
