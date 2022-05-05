import flask
from flask import Flask, request, render_template
from utils import get_ori_video, cache_display_images
from logger_utils import AILogger
import os

app = Flask(__name__)
app.secret_key = 'anyrandomkey456'
app.logger = None

@app.route("/")
def hello_world():
	if app.logger is None:
		return render_template('video.html')
	else:
		logger_profile = generate_video_list(app.logger.file_name)['logger_profile']
		return render_template('video.html', logger_file=app.logger.file_name,
							   logger_profile=logger_profile)

@app.route("/get_video_list", methods=['POST'])
def get_video_list():
	logfile_name = request.form.get('logfile_name')
	if logfile_name == '':
		return {'logger_profile': []}
	if os.path.exists(logfile_name):
		return generate_video_list(logfile_name)
	else:
		return {'err': 'no such file'}

def generate_video_list(logfile_name):
	app.logger = AILogger(logfile_name)
	app.logger.load(logfile_name)
	print('load from', logfile_name)
	video_list = list(app.logger.results.keys())[3:]
	ret_dict = []
	for video_name in video_list:
		frames = app.logger.results[video_name].keys()
		frame_num = len(frames)
		label_num = sum([len(app.logger.results[video_name][frame]['labels']) for frame in frames])
		pred_num = sum([len(app.logger.results[video_name][frame]['boxes']) for frame in frames])
		ret_dict.append({'video_name': video_name, 'frame_num': frame_num,
						 'label_num': label_num, 'pred_num': pred_num})
	return {'logger_profile': ret_dict}

@app.route("/load_video", methods=['GET'])
def load_video():
	video_name = request.args['video_name']
	if app.logger is None or video_name not in list(app.logger.results.keys())[3:]:
		return {'err':'no video name in this logger'}
	if 'min_display_conf' in request.args:
		min_display_conf = float(request.args['min_display_conf'])
	else:
		min_display_conf = 0.1
	dcm_scan_param_csv_file = '/home/pj-019468-si/BARDA_ID/Code/barda_lus/blines/image_proc_algorithm/Read_params_from_DICOMS/output/medstar_dicom_scan_params_updated.csv'
	dcm_cropped_imgs = get_ori_video(video_name, app.logger, min_display_conf=min_display_conf, img_type='jpg',
									 cls_names=['Con','SPC'], dcm_scan_param_csv_file=dcm_scan_param_csv_file)
	cache_dir = 'static/cache_imgs'
	img_src_dict = cache_display_images(dcm_cropped_imgs,video_name,app.logger.results[video_name].keys(),cache_dir)
	return {'img_src_dict': img_src_dict}
