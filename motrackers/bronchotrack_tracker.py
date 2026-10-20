import numpy as np
from scipy.optimize import linear_sum_assignment
from motrackers.utils.misc import iou_xywh as iou
from motrackers.track import BronchoTracklet
from motrackers.tracker import Tracker
from collections import OrderedDict
import torchvision.transforms as transforms

def assign_tracks2detection_bronchotrack(bbox_tracks, bbox_detections, embed_tracks, embed_detections, iou_threshold=0.3):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Args:
        bbox_tracks (numpy.ndarray): Bounding boxes of shape `(N, 4)` where `N` is number of objects already being tracked.
        bbox_detections (numpy.ndarray): Bounding boxes of shape `(M, 4)` where `M` is number of objects that are newly detected.
        iou_threshold (float): IOU threashold.

    Returns:
        tuple: Tuple contains the following elements in the given order:
            - matches (numpy.ndarray): Array of shape `(n, 2)` where `n` is number of pairs formed after matching tracks to detections. This is an array of tuples with each element as matched pair of indices`(track_index, detection_index)`.
            - unmatched_detections (numpy.ndarray): Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks (numpy.ndarray): Array of shape `(k,)` where `k` is the number of unmatched tracks.
    """

    if (bbox_tracks.size == 0) or (bbox_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    iou_matrix = np.zeros((bbox_tracks.shape[0], bbox_detections.shape[0]), dtype=np.float32)
    app_matrix = np.zeros((bbox_tracks.shape[0], bbox_detections.shape[0]), dtype=np.float32)
    for t in range(bbox_tracks.shape[0]):
        for d in range(bbox_detections.shape[0]):
            iou_matrix[t, d] = iou(bbox_tracks[t, :], bbox_detections[d, :])
            #embed_d = embed_detections[d].transpose()
            #embed_d = embed_d / embed_d.sum()
            #embed_t = embed_tracks[t].transpose()
            #embed_t = embed_t / embed_t.sum()
            #print(embed_d.sum(), embed_t.sum(),((embed_d*embed_t)/len(embed_d*embed_t)).sum())
            #app_matrix[t,d] = 1-np.matmul(embed_d.transpose(),embed_t)/len(embed_d)
            #print(iou_matrix[t,d],app_matrix[t,d])
            
    lambda_c = 0.5
    cost_matrix = iou_matrix #lambda_c*app_matrix + (1-lambda_c)*iou_matrix
    assigned_tracks, assigned_detections = linear_sum_assignment(-cost_matrix) #1-iou_matrix)
    #print(assigned_tracks,assigned_detections)
    unmatched_detections, unmatched_tracks = [], []

    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            #print('d',d,iou_matrix[:,d])
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):

        if t not in assigned_tracks:
            #print('t',t,iou_matrix[t,:])
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        #print(t,d,app_matrix[t,d],iou_matrix[t,d], cost_matrix[t,d])
        if iou_matrix[t,d]<0.3:#iou_threshold: #cost_matrix[t, d] > 0.9 and 
            
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class BronchoTracker(Tracker):
    """
    BronchoTrack - Multi object tracker.

    Args:
        tracker_output_format (str): Output format of the tracker.
        iou_threshold (float): Intersection over union minimum value.
        process_noise_scale (float or numpy.ndarray): Process noise covariance matrix of shape (3, 3)
            or covariance magnitude as scalar value.
        measurement_noise_scale (float or numpy.ndarray): Measurement noise covariance matrix of shape (1,)
            or covariance magnitude as scalar value.
        time_step (int or float): Time step for Kalman Filter.
    """

    def __init__(
            self,
            tracker_output_format='mot_challenge',
            iou_threshold=0.3,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            time_step=1,
            lambda_c=0.5
    ):
        self.time_step = time_step
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.kalman_trackers = OrderedDict()
        self.iou_threshold = iou_threshold
        self.lambda_c = lambda_c
        
        super().__init__(
            max_lost=5, tracker_output_format=tracker_output_format
        )

    def _add_track(self, frame_id, bbox, embed, **kwargs):
        self.tracks[self.next_track_id] = BronchoTracklet(self.next_track_id, frame_id, bbox, embed, process_noise_scale=1.0, measurement_noise_scale=1.0,kf_time_step=1)
        self.next_track_id += 1

    def _update_track(self, track_id, frame_id, bbox, embed, lost=0, **kwargs):
        self.tracks[track_id].update(
            frame_id, bbox, embed, lost=lost, **kwargs
        )

    def update(self,bboxes, crops):
        track_ids = list(self.tracks.keys())
        bbox_detections = np.array(bboxes)

        # First frame initialization
        if self.frame_count == 0:
            for i, bbox in enumerate(bbox_detections):

                transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]) 
                inp = transform(crops[i])
                inp = inp.unsqueeze(0).unsqueeze(2).to('cuda')
                #embed = self.net(inp).detach().cpu().numpy()
                #embed = (embed - embed.min()) / (embed.max() - embed.min())
                self._add_track(self.frame_count, bbox, None)
        else:
            # Motion Model
            bbox_tracks = []
            for track_id in track_ids:
                bb = self.tracks[track_id].bboxes[-1]#.predict()
                bbox_tracks.append(bb)#.astype(int))

            bbox_tracks = np.array(bbox_tracks)
            
            if len(bboxes) == 0:
                for i in range(len(bbox_tracks)):
                    track_id = track_ids[i]
                    bbox = bbox_tracks[i, :]
                    embed = self.tracks[track_id].embed
                    self._update_track(track_id, self.frame_count, bbox, embed, lost=1)
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)
            else:
                transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]) 
                

                matches, unmatched_detections, unmatched_tracks = assign_tracks2detection_bronchotrack(
                    bbox_tracks, bbox_detections,None,None, iou_threshold=self.iou_threshold)

                for i in range(matches.shape[0]):
                    t, d = matches[i, :]
                    track_id = track_ids[t]
                    bbox = bboxes[d, :]
                    alpha = 0.9
                    #updated_embed = alpha * embed_tracks[t] + (1-alpha) * embed_detections[d]
                   
                    self._update_track(track_id, self.frame_count, bbox, None, lost=0)

                for d in unmatched_detections:
                    bbox = bboxes[d, :]
                    self._add_track(self.frame_count, bbox, None)
                
                for t in unmatched_tracks:
                    track_id = track_ids[t]
                    bbox = self.tracks[track_id].bboxes[-1] #bbox_tracks[t, :]  
                    self._update_track(track_id, self.frame_count, bbox, None, lost=1)
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)
        output = self._get_tracks(self.tracks)
        #print(output)
        self.frame_count += 1
        return output