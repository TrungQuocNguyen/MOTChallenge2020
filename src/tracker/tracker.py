import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect #Faster RCNN model 

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes) #new_boxes: tensor Nx4, all boxes in a frame,new_scores: tensor N at initialization. 
		for i in range(num_new):#After that, new_boxes is list, new_scores is list 
			self.tracks.append(Track( #self.tracks is a list of Track. Each Track is different object in a frame. 
				new_boxes[i],
				new_scores[i],
				self.track_num + i #track id 
			))
		self.track_num += num_new#number of tracking objects only increase when we add new box( object that never occur before )

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame): #frame is dictionary of {img, img_path, gt, vis, seg_img(if load_seg = True)}
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# object detection. boxes and scores are predicted value computed by FasterRCNN
		boxes, scores = self.obj_detect.detect(frame['img'])#boxes: tensor Nx4, scores: tensor N, img: tensor CxHxW

		self.data_association(boxes, scores)

		# results
		for t in self.tracks: #get each Track in self.tracks
			if t.id not in self.results.keys(): #new track
				self.results[t.id] = {} #e.g: results[0] = {}, results = {0:{0: [x1,y1, x2,y2,scores]}, 1:{0:[x1,y1,x2,y2,scores]}, ..., id:{}}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])]) #np.array of 5 element: 4 for box and 1 for score

		self.im_index += 1

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box
		self.score = score
