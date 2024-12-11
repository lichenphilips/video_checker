from app import get_ori_video
from logger_utils import AILogger
#logfile_name='/mnt2/lx198/gary/relative_position_exps/pl_jsons/2mAP8AUC_test_video_varc_ori_coord_log_2024-09-23_065225_track_confp001_trckremoval_ioup1.json'
logfile_name='/mnt2/lx198/gary/relative_position_exps/val_video_ori_coord_log_2024-12-09_171652_track_confp001_trckremoval_ioup1_merged_with_PLpredictions.json'
logger = AILogger(logfile_name)
logger.load(logfile_name)
print('load from', logfile_name)
if 'class_names' in logger.results['platform']:
    cls_names = logger.results['platform']['class_names']
else:
    cls_names = ['Con', 'SPC', 'PE', 'AT']
video_name = '04-SI-0004-T01/C5-2U/image_148263715864063'
painted_imgs = get_ori_video(video_name, logger, show_frame_name=False,
                             box_display_method="min_display_conf",
                             box_display_method_variable=0.01,
                             img_type='dcm_npz', cls_names=cls_names)
