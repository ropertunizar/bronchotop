import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch_kmeans import KMeans
import torch
from sklearn.metrics import classification_report, accuracy_score
from bigtree import Node, get_subtree
from skimage.segmentation import slic, mark_boundaries 

def mask_overlay(image, mask, color=(1, 1, 1)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    #print(mask)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, :] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def define_airway():
    tra = Node('TRA', id=0)

    rmb = Node('RMB', id=1, parent=tra)

    trirul = Node('TriRUL', id=2, parent=rmb)
    bi = Node('BI', id=6, parent=rmb)
    
    rll7 = Node('RLL7', id=34, parent=bi)
    rb45 = Node('RB4+5', id=7, parent=bi)

    r1 = Node('R1', id=3, parent=trirul)
    r2 = Node('R2', id=4, parent=trirul)
    r3 = Node('R3', id=5, parent=trirul)

    trirll = Node('TriRLL', id=12, parent=rll7)
    r6 = Node('R6', id=10, parent=rll7)
    
    r4 = Node('R4', id=8, parent=rb45)
    r5 = Node('R5', id=9, parent=rb45)

    r7 = Node('R7', id=11, parent=trirll)
    r8 = Node('R8', id=14, parent=trirll)
    rb910 = Node('RB9+10', id=13, parent=trirll)

    r9 = Node('R9', id=15, parent=rb910)
    r10 = Node('R10', id=16, parent=rb910)

    lmb = Node('LMB', id=17, parent=tra)

    trilul = Node('TriLUL', id=18, parent=lmb)
    llb6 = Node('LLB6', id=26, parent=lmb)
    
    l6 = Node('L6', id=20, parent=llb6)
    trillb = Node('TriLLB', id=27, parent=llb6)

    l8 = Node('L8', id=21, parent=trillb)
    lb910 = Node('LB9+10', id=32, parent=trillb)

    l9 = Node('L9', id=28, parent=lb910)
    l10 = Node('L10', id=29, parent=lb910)

    lb12 = Node('LB1+2', id=35, parent=trilul)
    l3 = Node('L3', id=19, parent=trilul)
    lb45 = Node('LB4+5', id=25, parent=trilul)

    l1 = Node('L1', id=22, parent=lb12)
    l2 = Node('L2', id=23, parent=lb12)

    l4 = Node('L4', id=24, parent=lb45)
    l5 = Node('L5', id=30, parent=lb45)
    return tra

  
def detect_bboxes(img, area_thresh):
    bboxes = np.array([])
    crops = np.array([])
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # thresh = np.quantile(img_gray,0.1)    
    # coordinates = np.array(peak_local_max(255-cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), min_distance=30, labels=(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<=thresh).astype(np.uint8)))
    # segments = slic(img,n_segments=10,compactness=10)
    # vis = (mark_boundaries(img, segments)*255).astype(np.uint8)
    # num_segments = np.unique(segments)[-1]
    # seg = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    # num_segments = np.unique(segments)[-1]
    # for i in range(1, num_segments+1):
    #     for ck in range(len(coordinates)):
    #         if segments[coordinates[ck][0],coordinates[ck][1]]==i:
    #             seg[segments==i] = 1
    # firstTrack = False
    # contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # for cntr in contours:
    #     x,y,w,h = cv2.boundingRect(cntr)
    #     if not firstTrack:
    #                 bboxes = np.array([[x, y, w, h]], np.int32)
    #                 crop = img[x:x+w,y:y+h,:]
    #                 crop = cv2.resize(crop,(256,256))
    #                 crops = np.array([crop])
    #                 firstTrack = True     
    #     else:            
    #                 bboxes = np.append(bboxes, [[x, y, w, h]], axis=0)
    #                 crop = img[x:x+w,y:y+h,:]
    #                 crop = cv2.resize(crop,(256,256))
    #                 crops = np.append(crops, [crop],axis=0)
    # thresholded_w = seg    

    # thresh = np.quantile(img_gray,0.1)    
    # coordinates = peak_local_max(255-img_gray, min_distance=20, labels=(img_gray<=thresh).astype(np.uint8))

    # mask = np.zeros(img_gray.shape, dtype=bool)
    # mask[tuple(coordinates.T)] = True
    # markers, _ = ndi.label(mask)

    # thresholded_w = watershed(img_gray, markers, mask=(img_gray<=thresh).astype(np.uint8))
    # thresholded_w = (((thresholded_w -thresholded_w.min())/(thresholded_w.max()-thresholded_w.min()))*255).astype(np.uint8) 
    # (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresholded_w)
    # threshold_area = 0.02*(thresholded_w.shape[0])*(thresholded_w.shape[1])
    # firstTrack = False
    # for i in range(1, totalLabels):
    #     area = values[i, cv2.CC_STAT_AREA]
    #     x = values[i, cv2.CC_STAT_LEFT]
    #     y = values[i, cv2.CC_STAT_TOP]
    #     w = values[i, cv2.CC_STAT_WIDTH]
    #     h = values[i, cv2.CC_STAT_HEIGHT]
    #     if area > threshold_area: 
    #         thresholded_w[label_ids==i] = 255
    #         if not firstTrack:
    #                 bboxes = np.array([[x, y, w, h]], np.int32)
    #                 crop = img[x:x+w,y:y+h,:]
    #                 crop = cv2.resize(crop,(256,256))
    #                 crops = np.array([crop])
    #                 firstTrack = True     
    #         else:            
    #                 bboxes = np.append(bboxes, [[x, y, w, h]], axis=0)
    #                 crop = img[x:x+w,y:y+h,:]
    #                 crop = cv2.resize(crop,(256,256))
    #                 crops = np.append(crops, [crop],axis=0)
                    
    #     else:
    #         thresholded_w[label_ids==i] = 0

    device = 'cuda:0'
    img_rgb = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    original_shape = img.shape
    all_img = img.reshape((-1,1))

    all_pixels = torch.from_numpy(all_img[all_img>0].reshape((1,-1,1))).to(torch.float32).to(device)

    km = KMeans(n_clusters=5, random_state=0, n_init="auto")

    res = km(all_pixels)
    centers =  res.centers.squeeze(0)
    colors = torch.Tensor([50,100,150,200,255]).to(device)
    ind = torch.argsort(centers,axis=0) #np.argsort(np.array(centers),axis=0)
    indC = torch.argsort(ind,axis=0)
    colors = colors[indC]
    colors = torch.cat((torch.Tensor([0]).to(device),colors.flatten()),0)
    lab = torch.zeros(all_img.shape[0], dtype=torch.float32, device=device)

    lab[all_img[:,0]>0] = res.labels.squeeze(0).to(torch.float32) + 1 
    new_img = colors[lab.to(torch.int)] 
    new_img = new_img.reshape((original_shape))
    #new_img =  (((new_img-torch.min(new_img))/(torch.max(new_img)-torch.min(new_img)))*255)

    new_img = new_img.cpu().numpy().astype(np.uint8)
    #plt.subplot(121)
    #plt.imshow(img)
    #plt.subplot(122)
    #plt.imshow(new_img)
   
    # img_rgb = img.copy()
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # original_shape = img.shape
    # all_img = img.reshape((-1,1))

    # all_pixels = torch.from_numpy(all_img[all_img>0].reshape((1,-1,1))).to(torch.float32).to(device)

    # km = KMeans(n_clusters=5, random_state=0, n_init="auto")

    # res = km(all_pixels)
    # centers =  res.centers.squeeze(0)
    # colors = torch.Tensor([50,100,150,200,255]).to(device)
    # ind = torch.argsort(centers,axis=0) #np.argsort(np.array(centers),axis=0)
    # indC = torch.argsort(ind,axis=0)
    # colors = colors[indC]
    # colors = torch.cat((torch.Tensor([0]).to(device),colors.flatten()),0)
    # lab = torch.zeros(all_img.shape[0], dtype=torch.float32, device=device)

    # lab[all_img[:,0]>0] = res.labels.squeeze(0).to(torch.float32) + 1 
    # new_img = colors[lab.to(torch.int)] 
    # new_img = new_img.reshape((original_shape))
    # new_img =  (((new_img-torch.min(new_img))/(torch.max(new_img)-torch.min(new_img)))*255)

    # new_img = new_img.cpu().numpy().astype(np.uint8)
    _, thresholded = cv2.threshold(new_img,50,255,cv2.THRESH_BINARY_INV) 

    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresholded)
    threshold_area = area_thresh*(thresholded.shape[0])*(thresholded.shape[1]) 
    thresholded_w = np.zeros((thresholded.shape[0],thresholded.shape[1]),dtype=np.uint8)
    firstTrack = False

    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        x = values[i, cv2.CC_STAT_LEFT]
        y = values[i, cv2.CC_STAT_TOP]
        w = values[i, cv2.CC_STAT_WIDTH]
        h = values[i, cv2.CC_STAT_HEIGHT]
        #print(threshold_area,area, [x, y, w, h])
        if area > threshold_area: 
            thresholded_w[label_ids==i] = 255
            if not firstTrack:
                bboxes = np.array([[x, y, w, h]], np.int32)
                crop = img_rgb[x:x+w,y:y+h,:]
                crop = cv2.resize(crop,(256,256))
                crops = np.array([crop])
                firstTrack = True     
            else:            
                bboxes = np.append(bboxes, [[x, y, w, h]], axis=0)
                crop = img_rgb[x:x+w,y:y+h,:]
                crop = cv2.resize(crop,(256,256))
                crops = np.append(crops, [crop],axis=0)
        else:
             thresholded_w[label_ids==i] = 0
    #plt.show()
    return bboxes, crops, thresholded_w 

