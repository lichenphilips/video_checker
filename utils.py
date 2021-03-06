import cv2
import numpy as np
import copy
import os
import pydicom
import pandas as pd
import imageio

def paint_img_with_bb(image ,boxes ,labels=[] ,title=None ,min_display_conf=0 ,cls_names=None):
    int_max =  1  # np.max(image)
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    linewidth = 2
    # include ground truth labels
    paint_boxes = copy.deepcopy(labels)
    # include preditions
    for box in boxes:
        conf = box['conf']
        if conf > min_display_conf:
            paint_boxes.append(box)

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
        image = cv2.rectangle(image, start_point, end_point, edgecolor, linewidth)
        if cls != 0:
            if cls_names is None:
                cls_label = 'cls:%d' % cls
            else:
                cls_label = cls_names[int(cls)]
            image = cv2.putText(image, cls_label, start_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=edgecolor, thickness=2)

    if title:
        if len(title) < 30:
            image = cv2.putText(image, title, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(int_max, int_max, int_max), thickness=1)
        elif len(title) < 45:
            image = cv2.putText(image, title, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=(int_max, int_max, int_max), thickness=1)
        else:
            title_splits = title.split('/')
            for si, title_split in enumerate(title_splits):
                image = cv2.putText(image, title_split, (30, 30 + (si * 20)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                                    color=(int_max, int_max, int_max), thickness=1)

    return image

def get_ori_frame(video_name, frame_id, logger, show_labels=True, show_preds=True,
                  min_display_conf=0, img_type='jpg', cls_names=None):
    data_dir_prefix = logger.results['platform']['data_dir_prefix']
    frame_name = '%d' % (frame_id)
    if img_type == 'jpg':
        img_filename = data_dir_prefix + video_name + '_%02d.jpg' % (frame_id)
        if not os.path.exists(img_filename):
            img_filename = img_filename.replace('/no_consolidation_area', '/consolidation_area').replace(
                '/consolidation_area', '/no_consolidation_area')
            if not os.path.exists(img_filename):
                print('img_filename not exist', img_filename)
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_filename)
        img = cv2.imread(img_filename)
        img = img / np.max(img)
    else:
        raise ValueError('undefined')
    if show_preds:
        boxes = logger.results[video_name][frame_name]['boxes']
    else:
        boxes = []
    if show_labels:
        labels = logger.results[video_name][frame_name]['labels']
    else:
        labels = []
    img_bb = paint_img_with_bb(img, boxes, labels, video_name + '_' + frame_name, min_display_conf, cls_names)
    return img_bb


def get_ori_video(video_name, logger, show_labels=True, show_preds=True,
                  min_display_conf=0, img_type='jpg', cls_names=None, dcm_scan_param_csv_file=None):
    imgs = []
    frames = sorted(logger.results[video_name], key=lambda x: int(x))
    if img_type == 'jpg':
        for frame_name in frames:
            frame_id = int(frame_name)
            img_bb = get_ori_frame(video_name, frame_id, logger, show_labels, show_preds, min_display_conf, img_type,
                                   cls_names)
            imgs.append(img_bb)
    elif img_type == 'dcm':
        data_dir_prefix = logger.results['platform']['data_dir_prefix']
        img_filename = data_dir_prefix + video_name + '.dcm'
        dcm_img = pydicom.read_file(img_filename).pixel_array[:, :, :, 0]
        dcm_cropped_imgs, img_offset = crop_dcm_img(dcm_img, video_name + '.dcm', dcm_scan_param_csv_file)
        dcm_cropped_imgs = dcm_cropped_imgs / np.max(dcm_cropped_imgs)
        imgs = []
        for frame_name in logger.results[video_name]:
            frame_id = int(frame_name)
            if show_preds:
                boxes = logger.results[video_name][frame_name]['boxes']
            else:
                boxes = []
            if show_labels:
                labels = logger.results[video_name][frame_name]['labels']
            else:
                labels = []
            img_bb = paint_img_with_bb(dcm_cropped_imgs[frame_id], boxes, labels, video_name + '_' + frame_name,
                                       min_display_conf, cls_names)
            imgs.append(img_bb)
    else:
        raise ValueError('undefined')
    return imgs

# remove unnecessary info outside scan region. Use dcm_scan_param_csv to locate boundary
def crop_dcm_img(dcm_img, path, dcm_scan_param_csv_file):
    dcm_scan_param_csv = pd.read_csv(dcm_scan_param_csv_file)
    # dcm_img: frame, rows, columns
    path_base = path.split('/')[-1]
    scan_param = dcm_scan_param_csv[dcm_scan_param_csv['filename'] == path_base]
    if len(scan_param) == 0:
        raise ValueError(path, 'dcm not found in scan param csv')
    if len(scan_param) > 1:
        print('more than one dcm in scan param csv')
    geo = scan_param.iloc[0]
    # convert string to tuple
    for column in ['apex', 'angle_rad', 'slopes', 'pixel_size_cm', 'rows_cols', 'top_left', 'top_right',
                   'bottom_left', 'bottom_right']:
        geo[column] = [float(i) for i in geo[column][1:-1].split(',')]
    crop_mask = np.zeros([geo['rows'], geo['columns']], dtype=np.uint8)
    apex = tuple(np.round(geo['apex']).astype(dtype=int))
    r1 = int(round(geo['radius_pixel']))
    r2 = int(round(geo['apex_to_depth_pixel']))
    a1 = geo['angle_rad'][0] * 180 / np.pi
    a2 = -geo['angle_rad'][1] * 180 / np.pi
    cv2.ellipse(crop_mask, apex, (r2, r2), 90, a1, a2, color=1, thickness=-1)
    cv2.ellipse(crop_mask, apex, (r1, r1), 90, a1, a2, color=0, thickness=-1)

    masked_img = np.multiply(dcm_img, np.repeat(crop_mask[None], dcm_img.shape[0], axis=0))
    image_cropped = masked_img[:, geo['top']:geo['bottom'], geo['left']:geo['right']]
    return image_cropped, (geo['left'], geo['top'],
                           geo['columns'], geo['rows'],
                           geo['right'] - geo['left'], geo['bottom'] - geo['top'])

#save predicted image to server cache
def cache_display_images(dcm_cropped_imgs,video_name,frame_names,cache_dir,dir_path):
    img_src_dict = {}
    if not os.path.exists(dir_path+'/'+cache_dir+'/'+video_name):
        os.makedirs(dir_path+'/'+cache_dir+'/'+video_name)
    img_max_int = np.max(dcm_cropped_imgs)
    imgs_uint8 = []
    for fi, frame_name in enumerate(frame_names):
        cache_filename = cache_dir+'/'+video_name+'/'+frame_name+'.jpg'
        img = dcm_cropped_imgs[fi]
        img_uint8 = (img / img_max_int * 255).astype(np.uint8)
        cv2.imwrite(dir_path+'/'+cache_filename, img_uint8)
        imgs_uint8.append(img_uint8)
        img_src_dict[frame_name] = cache_filename
    gif_filename = dir_path + '/' + cache_dir + '/' + video_name + '/' + (video_name).replace('/', '_') + '.gif'
    print('save gif', gif_filename)
    imageio.mimsave(gif_filename, imgs_uint8, fps=5)
    return img_src_dict