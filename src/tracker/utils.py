#########################################
# Still ugly file with helper functions #
#########################################

import os
import random
from collections import defaultdict
from os import path as osp

#import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
from cycler import cycler as cy
from scipy.interpolate import interp1d
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

def plot_sequence(tracks, db, first_n_frames=None):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        
        tracks: Tracker.results
        db: MOT16Sequence object 
    """

    # print("[*] Plotting whole sequence to {}".format(output_dir))

    # if not osp.exists(output_dir):
    # 	os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        img = v['img'].mul(255).permute(1, 2, 0).byte().numpy() #np. array size HxWxC
        width, height, _ = img.shape

        dpi = 96
        fig, ax = plt.subplots(1, dpi=dpi)
        fig.set_size_inches(width / dpi, height / dpi)
        ax.set_axis_off()
        ax.imshow(img)

        for j, t in tracks.items():
            if i in t.keys(): # if object of track t has frame i ( aka if object of track t occur in frame i)
                t_i = t[i] #coordinate of bounding box of object t (track t, each object corresponds to a track) 
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j] #each 
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=6, ha='center', va='center')

        plt.axis('off')
        #----------------------------------------------------------------------------------
        #plt.gca().set_axis_off()
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        #plt.margins(0,0)
        #plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #------------------------------------------------------------------------------------
        #plt.tight_layout()
        plt.show()
        
        #fig.savefig('/content/MOTChallenge_image/file%02d.png' % i, dpi=100, bbox_inches='tight')
        # plt.close()

        if first_n_frames is not None and first_n_frames - 1 == i:
            break



#run through each frame. With each frame, check if a track occurs in that frame or not 
def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)
    
    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items(): #gt is a dictionary of {box_id: box coordinates(array size 4)}
                gt_ids.append(gt_id) #gt_ids: ids of all groundtruth boxes in that frame, len N
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0) #gt_boxes: array Nx4, N is number of groundtruth box 
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1) #array Nx4
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items(): #frames = { 0: array[ ], 1:  array[ ], 2:array[ ] }. frames: all frames track_id occurs
            if i in frames: #if that object (with track_id) appears in that frame i
                track_ids.append(track_id) #len K
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4]) #only take x1,y1,x2,y2. Omit score 
        #track_ids: ids of all predicted boxes belong to that frame
        #track_boxes: boxes(coordinates) of all ids that belong to that frame 
        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)#size Kx4, K: number of predicted boxes
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)#array NxK, N: number of groundtruth, K: number of predicted

        mot_accum.update(
            gt_ids,
            track_ids, #gt_ids: ids of all groundtruth boxes, len N, track_ids: ids of all predicted boxes, len K
            distance)#distance: array NxK. 

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)


def evaluate_obj_detect(model, data_loader):
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(data_loader): #imgs is Tensor of size batchsize xCxHxW, targets is list of dictionary of {boxes, labels, image_id, area, iscrowd, visibilities}
        imgs = [img.to(device) for img in imgs] #Turn Tensor into list of tensor to feed in FasterRCNN model

        with torch.no_grad():
            preds = model(imgs) # preds is List[Dict[Tensor]], one for each input image. dict of {boxes, labels, scores }. See pytorch FasterRCNN

        for pred, target in zip(preds, targets): #each pred is dict, each target is dict 
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(), #Tensor Nx4
                                                  'scores': pred['scores'].cpu()} #Tensor N

    data_loader.dataset.print_eval(results)


def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def collate_fn(batch):
    img_list = []
    gt = {}
    img_path = []
    vis = {}
    seg_img = []
    for sample in batch: 
      img_list.append(sample['img'])
      img_path.append(sample['img_path'])
      if 'seg_img' in sample: 
        seg_img.append(torch.tensor(sample['seg_img']))
      #tensor_batch['img'] = torch.stack([sample['img']for sample in batch])
      for key in sample['gt'].keys():
        if key not in gt.keys():
          gt[key] = [] 
        gt[key].append(torch.from_numpy(sample['gt'][key]))
      for key in sample['vis'].keys(): 
        if key not in vis.keys():
          vis[key] = [] 
        vis[key].append(sample['vis'][key])
    img = torch.stack(img_list)
    for key in gt.keys(): 
      gt[key] = torch.stack(gt[key])
    for key in vis.keys(): 
      vis[key] = torch.tensor(vis[key])
    
    
    batch_frames = {}
    batch_frames['img'] = img
    batch_frames['gt'] = gt
    batch_frames['img_path'] = img_path
    batch_frames['vis'] = vis
    if len(seg_img): 
      seg_img = torch.stack(seg_img)
      batch_frames['seg_img'] = seg_img
    return batch_frames
