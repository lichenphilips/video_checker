# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:25:18 2021

@author: usd15988
"""
import json
import platform
from datetime import datetime
import hashlib
from json import encoder
import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2

encoder.FLOAT_REPR = lambda o: format(o, '.2f')


class AILogger():
    ''' Logger class for logging classification and/or detection results in muli-frame videos'''

    def __init__(self, file_name=None, opt=None):
        self.file_name = file_name
        self.video_name = ""

        platform_info = {}
        platform_info["machine"] = platform.machine()
        platform_info["system"] = platform.system()
        platform_info["release"] = platform.release()
        platform_info["python_version"] = platform.python_version()

        self.results = {}
        self.results["creation_date_time"] = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.results["platform"] = platform_info

        if opt:
            self.results["experiment"] = opt.experiment
            self.results["data_config"] = opt.data_config
            self.results["model_def"] = opt.model_def
            self.results["model_type"] = opt.model_type
            self.results["epoch"] = opt.start_epoch
            self.results["test_batch_size"] = opt.batch_size
            self.results["conf_thres"] = opt.conf_thres[0]
            self.results["iou_thres"] = opt.iou_thres[0]
            self.results["nms_thres"] = opt.nms_thres
            self.results["num_classes"] = opt.num_classes
            self.results["img_channels"] = opt.img_channels
            self.results["img_size"] = opt.img_size
            self.results["img_type"] = opt.img_type
            self.results["multiscale"] = opt.multiscale
            self.results["align_top"] = opt.align_top
            self.results["flip_lr"] = opt.flip_lr
            self.results["rescale_to_orig_size"] = opt.rescale_to_original_image_size

        self.results["cascade"] = {}

    @staticmethod
    def __calculate_sha256(file_name):
        with open(file_name, "rb") as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()
        return file_hash

    def set_detector_info(self, detection_params):
        detector = {}
        detector["model"] = detection_params['model_path']
        detector["model_sha256"] = self.__calculate_sha256(detection_params['model_path'])
        detector["confidence_threshold"] = detection_params['confidence_threshold']
        self.results["cascade"]["detector"] = detector

    def set_video_name(self, video_name):
        '''Current video name'''
        self.video_name = video_name

    def get_video_name(self):
        '''Return current video name'''
        return self.video_name

    def set_video_pix_per_cm(self, video_name, pix_per_cm):
        ''' set the pix_per_cm assumed/calcualted for this video '''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        self.results[video_name]['pix_per_cm'] = pix_per_cm

    def set_video_cropping_start_point(self, video_name, cropping_start_point_row_col):
        ''' set the cropping start point (upper left corner of cropped area in original full frame), in [row, col] format for this video '''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        self.results[video_name]['cropping_start_point_row_col'] = cropping_start_point_row_col

    def set_video_cropping_width_height(self, video_name, cropping_width_height):
        ''' set the cropping width and height in [width, height] format for this video '''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        self.results[video_name]['cropping_width_height'] = cropping_width_height

    def set_video_number_of_frames(self, video_name, number_of_frames):
        ''' set the total number of frames for this video '''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        self.results[video_name]['number_of_frames'] = number_of_frames

    def add_label(self, video_name, frame_name, label_box, label_class):
        ''' add label (box coords) for a frame in a video to the results. Instead of
            an invdividual label, can also add a list of labels'''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        if not frame_name in self.results[video_name]:
            self.results[video_name][frame_name] = {}  # For each frame - a dict
        if not 'num_labels' in self.results[video_name][frame_name]:
            self.results[video_name][frame_name]['num_labels'] = -1
        if not 'labels' in self.results[video_name][frame_name]:
            self.results[video_name][frame_name]['labels'] = []  # start an empty list for boxes
        if type(label_box) is list:
            these_boxes = [{'coords': coords, 'class': float(cls)} for (coords, cls) in zip(label_box, label_class)]
            self.results[video_name][frame_name]['num_labels'] = len(label_box)
            self.results[video_name][frame_name]['labels'].extend(these_boxes)
        else:
            this_box = {'coords': label_box, 'class': float(label_class)}
            self.results[video_name][frame_name]['num_labels'] = 1
            self.results[video_name][frame_name]['labels'].append(this_box)

    def add_classification_result(self, video_name, frame_name, result):
        result = float(result)
        ''' add a classification result (confidence score) for a frame in a video to the results '''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        if not frame_name in self.results[video_name]:
            self.results[video_name][frame_name] = {}  # For each frame - a dict
        self.results[video_name][frame_name]['conf'] = result

    def add_detection_result(self, video_name, frame_name, box, box_confidence):
        ''' add a detection result (box coords and box confidence) for a frame in a video to the results. Instead of
            an invdividual box and box_confidence, can also add a list of boxes and list of box_confidence scores'''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        if not frame_name in self.results[video_name]:
            self.results[video_name][frame_name] = {}  # For each frame - a dict
        if not 'boxes' in self.results[video_name][frame_name]:
            self.results[video_name][frame_name]['boxes'] = []  # start an empty list for boxes
        if type(box_confidence) is list:
            these_boxes = [{'coords': box[idx], 'conf': float(bc)} for idx, bc in enumerate(box_confidence)]
            self.results[video_name][frame_name]['boxes'].extend(these_boxes)
        else:
            this_box = {'coords': box, 'conf': float(box_confidence)}
            self.results[video_name][frame_name]['boxes'].append(this_box)

    def add_detection_result_with_class(self, video_name, frame_name, box, box_confidence, box_class):
        ''' add a detection result (box coords and box confidence) for a frame in a video to the results. Instead of
            an individual detection, can also add a list of detections'''
        if not video_name in self.results:
            self.results[video_name] = {}  # For each video - a dict
        if not frame_name in self.results[video_name]:
            self.results[video_name][frame_name] = {}  # For each frame - a dict
        if not 'num_boxes' in self.results[video_name][frame_name]:
            self.results[video_name][frame_name]['num_boxes'] = -1
        if not 'boxes' in self.results[video_name][frame_name]:
            self.results[video_name][frame_name]['boxes'] = []  # start an empty list for boxes
        if type(box) is list:
            these_boxes = [{'coords': coords, 'conf': float(conf), 'class': float(cls)} for (coords, conf, cls) in
                           zip(box, box_confidence, box_class)]
            self.results[video_name][frame_name]['num_boxes'] = len(box)
            self.results[video_name][frame_name]['boxes'].extend(these_boxes)
        else:
            this_box = {'coords': box, 'conf': float(box_confidence), 'class': float(box_class)}
            self.results[video_name][frame_name]['num_boxes'] = 1
            self.results[video_name][frame_name]['boxes'].append(this_box)

    '''  
    def add_detections_from_list(self, img_paths, outputs):
        for i, (img_path_i, out_i) in enumerate(zip(img_paths, outputs)):
            if isinstance(img_path_i, list):
                img_path_i = img_path_i[0] # if len(img_path_i) > 0:
            video_name = 'image_' + img_path_i.split('image_')[-1].split('.jpg')[0].split('_')[0]
            frame_no = int(img_path_i.split('_')[-1].split('.jpg')[0])
            if out_i is None:
                self.add_detection_result_with_class(video_name, 
                                                     frame_no,
                                                     box=[],
                                                     box_confidence=[],
                                                     box_class=[]
                                                    )
            else:
                self.add_detection_result_with_class(video_name, 
                                                     frame_no,
                                                     box=out_i[:, :4].tolist(),
                                                     box_confidence=out_i[:, 4].tolist(),
                                                     box_class=out_i[:, 6].tolist()
                                                    )

    def add_detections_and_labels_from_list(self, img_paths, outputs, labels):
        print(img_paths)
        print(labels)
        for i, (img_path_i, out_i, label_i) in enumerate(zip(img_paths, outputs, labels)):
            print(i)
            print(img_path_i)
            print(label_i)
            if isinstance(img_path_i, list):
                img_path_i = img_path_i[0] # if len(img_path_i) > 0:
            video_name = 'image_' + img_path_i.split('image_')[-1].split('.jpg')[0].split('_')[0]
            frame_no = int(img_path_i.split('_')[-1].split('.jpg')[0])

            self.add_label(video_name,
                           frame_no,
                           label_box=label_i[:, 2:].tolist(),
                           label_class=label_i[:, 1].tolist()
                          )

            if out_i is None:
                self.add_detection_result_with_class(video_name, 
                                                     frame_no,
                                                     box=[],
                                                     box_confidence=[],
                                                     box_class=[]
                                                    )
            else:
                self.add_detection_result_with_class(video_name, 
                                                     frame_no,
                                                     box=out_i[:, :4].tolist(),
                                                     box_confidence=out_i[:, 4].tolist(),
                                                     box_class=out_i[:, 6].tolist()
                                                    )
    '''

    def add_detections_and_labels_from_list(self, img_paths, outputs, labels, add_labels_to_log_file=False):
        ''' add a detection result (box coords, confidence, and class) for a frame in a video to the results.
            Optionally, can also add the label result (box coords and class).'''
        for i, (img_path_i, out_i) in enumerate(zip(img_paths, outputs)):
            if isinstance(img_path_i, list):
                img_path_i = img_path_i[0]  # if len(img_path_i) > 0:
            if '.jpg' in img_path_i:
                # video_name = 'image_' + img_path_i.split('image_')[-1].split('.jpg')[0].split('_')[0]
                # frame_no = int(img_path_i.split('_')[-1].split('.jpg')[0])
                video_name = img_path_i[:-len(img_path_i.split('_')[-1]) - 1]#.split('/')[-1]
                frame_no = str(int(img_path_i.split('_')[-1].split('.')[0]))
            elif '.dcm' in img_path_i:
                video_name = img_path_i.split('.dcm:')[0]
                frame_no = str(int(img_path_i.split('_')[-1].split('.dcm:')[-1]))
            else:
                raise ValueError('Unknown img type, do not know how to separate frame no')
            if add_labels_to_log_file:
                idx = [lb == i for lb in labels[:, 0]]
                label_i = labels[idx, :]
                self.add_label(video_name,
                               frame_no,
                               label_box=label_i[:, 2:].tolist(),
                               label_class=label_i[:, 1].tolist()
                               )
            if out_i is None:
                self.add_detection_result_with_class(video_name,
                                                     frame_no,
                                                     box=[],
                                                     box_confidence=[],
                                                     box_class=[]
                                                     )
            else:
                self.add_detection_result_with_class(video_name,
                                                     frame_no,
                                                     box=out_i[:, :4].tolist(),
                                                     box_confidence=out_i[:, 4].tolist(),
                                                     box_class=out_i[:, 5].tolist()
                                                     )

    def get_results(self):
        ''' get the entire results dict '''
        return self.results

    def get_video_results(self, video_name):
        ''' get the results dict for a particular video '''
        if video_name in self.results:
            return self.results[video_name]
        return None

    def get_frame_results(self, video_name, frame_name):
        ''' get the results dict for a particular video '''
        if video_name in self.results:
            if frame_name in self.results[video_name]:
                return self.results[video_name][frame_name]
        return None

    def get_video_frames(self, video_name):
        ''' get a list of video frame names from the dict of a particular video '''
        if video_name in self.results:
            video_results = self.results[video_name]
            frame_names = list(video_results.keys())
            frame_names.sort()
            return frame_names
        return None

    def get_video_frames_and_confidences(self, video_name):
        ''' get a list of video frame names and corresponding confidence values from the dict of a particular video (only returns frames for which ['conf'] is available) '''
        if video_name in self.results:
            video_results = self.results[video_name]
            frame_names = [k for k in video_results.keys() if k.isnumeric()]
            frame_names.sort(key=int)
            frame_confidences = [video_results[k]['conf'] for k in frame_names]
            return frame_names, frame_confidences
        return None, None

    def get_video_frames_and_boxes(self, video_name):
        ''' get a list of video frame names and corresponding boxes from the dict of a particular video (only returns frames for which ['boxes'] is available) '''
        if video_name in self.results:
            video_results = self.results[video_name]
            # frame_names= [k for k in video_results.keys() if 'boxes' in video_results[str(k)]]
            frame_names = []
            if type(video_results) == dict:
                for key in video_results.keys():
                    if type(video_results[key]) == dict:
                        if 'boxes' in video_results[key]:
                            frame_names.append(key)
            frame_names.sort(key=int)
            frame_boxlists = [video_results[k]['boxes'] for k in frame_names]
            return frame_names, frame_boxlists
        return None, None

        # examples
        # results['video1']['frame1']['classification_confidence'] = 0.2
        # results['video2']['frame1']['detection_box'][i]['coordinates'] = [10,20,400,500]
        # results['video2']['frame1']['detection_box'][i]['confidence'] = 0.3

    def get_all_video_frames_and_boxes(self, normalize=[]):
        ''' get a list of all video frame names and corresponding boxes from the dict'''
        frames = []
        boxes = []
        for video_name in self.results:
            video_base_name = video_name.split('.')[0]
            frame_names, frame_boxlists = self.get_video_frames_and_boxes(video_name)
            for frame_name, frame_boxlist in zip(frame_names, frame_boxlists):
                video_frame_no = f"{video_base_name}_{int(frame_name):02}"
                frame_boxes = []
                for box in frame_boxlist:
                    coords = box['coords']
                    if normalize:
                        coords[0] /= normalize[0]
                        coords[1] /= normalize[1]
                        coords[2] /= normalize[0]
                        coords[3] /= normalize[1]
                    coords.append(box['conf'])
                    coords.append(0)
                    frame_boxes.append(coords)
                frames.append(video_frame_no)
                boxes.append(frame_boxes)

        assert (len(frames) == len(boxes))
        return frames, boxes

    def get_video_frames_and_boxes_and_labels(self, video_name):
        ''' get a list of video frame names and corresponding boxes from the dict of a particular video (only returns frames for which ['boxes'] is available) '''
        if video_name in self.results:
            video_results = self.results[video_name]
            frame_names = []
            if type(video_results) == dict:
                for key in video_results.keys():
                    if type(video_results[key]) == dict:
                        if 'boxes' in video_results[key]:
                            frame_names.append(key)
            frame_names.sort(key=int)
            frame_boxlists = [video_results[k]['boxes'] for k in frame_names]
            label_boxlists = [video_results[k]['labels'] for k in frame_names]
            return frame_names, frame_boxlists, label_boxlists
        return None, None

    def get_all_video_frames_and_boxes_and_labels(self, normalize=[]):
        ''' get a list of all video frame names and corresponding prediction and label boxes from the dict'''
        frames_all = []
        pred_all = []
        labels_all = []

        for video_name in self.results:
            video_base_name = video_name.split('.')[0]
            frame_names, frame_boxlists, label_boxlists = self.get_video_frames_and_boxes_and_labels(video_name)
            for frame_name, frame_boxlist, label_boxlist in zip(frame_names, frame_boxlists, label_boxlists):
                frames_all.append(f"{video_base_name}_{int(frame_name):02}")

                boxes = []
                for box in label_boxlist:
                    coords = box['coords']
                    if normalize:
                        coords[0] /= normalize[0]
                        coords[1] /= normalize[1]
                        coords[2] /= normalize[0]
                        coords[3] /= normalize[1]
                    coords.append(box['class'])
                    boxes.append(coords)
                labels_all.append(boxes)

                boxes = []
                for box in frame_boxlist:
                    coords = box['coords']
                    if normalize:
                        coords[0] /= normalize[0]
                        coords[1] /= normalize[1]
                        coords[2] /= normalize[0]
                        coords[3] /= normalize[1]
                    coords.append(box['conf'])
                    coords.append(box['class'])
                    boxes.append(coords)
                pred_all.append(boxes)

        assert (len(frames_all) == len(labels_all))
        assert (len(frames_all) == len(pred_all))
        return frames_all, pred_all, labels_all

    def save(self, fn=None):
        if fn==None:
            fn = self.file_name
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent='\t')

    def load(self, fn=None):
        if fn==None:
            fn = self.file_name
        f = open(fn)
        self.results = json.load(f)
        f.close()

    def plt_frame(self, video_name, frame_no, img, min_display_conf = 0.3):
        # img_pad: original image size, padded to square
        if video_name not in self.results:
            raise ValueError('video not exist')
        if frame_no not in self.results[video_name]:
            raise ValueError('not exist frame')

        labels = self.results[video_name][frame_no]['labels']
        boxes = self.results[video_name][frame_no]['boxes']

        int_max = 1  # np.max(image)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)

        linewidth = 3
        # include ground truth labels
        paint_boxes = copy.deepcopy(labels)
        # include preditions
        for box in boxes:
            conf = box['conf']
            if conf > min_display_conf:
                paint_boxes.append(box)

        img_bb = img
        for box in paint_boxes:
            labels_xyxy = box['coords']
            cls = box['class']
            if 'conf' not in box:
                # labels
                edgecolor = [int_max, int_max, int_max]
            else:
                edgecolor = [box['conf'] * int_max, 0, (1 - box['conf']) * int_max]
                # edgecolor = colors[int(cls)]

            start_point = (int(round(labels_xyxy[0])), int(round(labels_xyxy[1])))
            end_point = (int(round(labels_xyxy[2])), int(round(labels_xyxy[3])))
            img_bb = cv2.rectangle(img, start_point, end_point, edgecolor, linewidth)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_bb)
        plt.show()


if __name__ == "__main__":
    AIL = AI_logger()
    AIL.set_video_pix_per_cm('image_1111111', 55)
    AIL.set_video_cropping_start_point('image_1111111', [300, 100])
    AIL.set_video_cropping_start_point('image_1111111', (300, 100))
    AIL.set_video_cropping_width_height('image_1111111', [500, 400])
    AIL.set_video_number_of_frames('image_1111111', 61)

    AIL.add_classification_result('image_1111111', '02', 0.2)
    AIL.add_classification_result('image_1111111', '01', 0.3)
    AIL.add_classification_result('image_2222222', '03', 0.13)
    AIL.add_detection_result('image_2222222', '03', [10, 20, 100, 200], 0.55)
    AIL.add_detection_result('image_2222222', '04', [11, 21, 110, 201], 0.56)
    AIL.add_detection_result('image_2222222', '04', [12, 22, 112, 202], 0.52)

    AIL2 = AI_logger()
    AIL2.add_classification_result('image_1111111', '02', 0.2)
    AIL2.add_classification_result('image_1111111', '01', 0.3)
    AIL2.add_classification_result('image_2222222', '03', 0.13)
    AIL2.add_detection_result('image_2222222', '03', [10, 20, 100, 200], 0.55)
    AIL2.add_detection_result('image_2222222', '04', [[11, 21, 110, 201], [12, 22, 112, 202]], [0.56, .52])
    fn = r'C:/temp/test_log2.json'
    AIL2.save(fn)
    print("AIL2 = ", AIL2.get_results())

    fn = r'C:/temp/test_log1.json'
    AIL.save(fn)
    print("AIL  = ", AIL.get_results())

    AIL2 = AI_logger()
    AIL2.load(fn)
    print("\nAIL2 = ", AIL2.get_results())

    print("\n")
    print(AIL2.get_video_results('image_1111111'))

    print("\n")
    frame_list = AIL2.get_video_frames('image_2222222')
    print(frame_list)

    print("\n")
    frame_list, frame_confidences = AIL2.get_video_frames_and_boxes('image_2222222')
    print(frame_list)
    print(frame_confidences)
