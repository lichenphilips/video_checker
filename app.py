import flask
from flask import Flask, request, render_template
from utils import get_ori_video, cache_display_images
from logger_utils import AILogger
import os
import shutil

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
		clear_cache()
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
		label_num = sum([len(app.logger.results[video_name][frame]['labels']) for frame in frames if 'labels' in app.logger.results[video_name][frame]])
		pred_num = sum([len(app.logger.results[video_name][frame]['boxes']) for frame in frames])
		ret_dict.append({'video_name': video_name, 'frame_num': frame_num,
						 'label_num': label_num, 'pred_num': pred_num})
	return {'logger_profile': ret_dict}

@app.route("/load_video", methods=['GET'])
def load_video():
	video_name = request.args['video_name']
	if app.logger is None or video_name not in list(app.logger.results.keys())[3:]:
		return {'err':video_name+' video name not in this logger'}
	box_display_method = None
	if 'min_display_conf' in request.args:
		box_display_method = 'min_display_conf'
		box_display_method_variable = float(request.args['min_display_conf'])
	elif 'top_N_frame_conf' in request.args:
		box_display_method = 'top_N_frame_conf'
		box_display_method_variable = int(request.args['top_N_frame_conf'])
	elif 'top_conf_frames_std' in request.args:
		box_display_method = 'top_conf_frames_std'
		box_display_method_variable = int(request.args['top_conf_frames_std'])
	else:
		box_display_method = 'min_display_conf'
		box_display_method_variable = 0.1
	frame_names = sorted(app.logger.results[video_name].keys(),key=lambda x:int(x))
	
	#test whether gif exists
	dir_path = os.path.dirname(os.path.realpath(__file__))
	cache_dir = 'static/cache_imgs'
	current_video_cache_path = cache_dir + '/' + video_name + '/' + box_display_method + '/%.3f'%(box_display_method_variable)
	gif_filename =  dir_path + '/' + current_video_cache_path + '/' + (video_name).replace('/', '_') + '.gif'

	#with open('video_checker_log.txt','w') as fp:
	#	fp.write(box_display_method+' '+str(box_display_method_variable))

	if os.path.exists(gif_filename):
		print('found cache',gif_filename)
		#construct img_src_dict by image
		img_src_dict = {}
		for fi, frame_name in enumerate(frame_names):
			cache_filename = current_video_cache_path+'/'+frame_name+'.jpg'
			if os.path.exists(dir_path+'/'+cache_filename):
				img_src_dict[frame_name] = cache_filename
	else:
		print('create cache',gif_filename)
		if 'data_dir_prefix' in app.logger.results['platform']:
			data_dir_prefix = app.logger.results['platform']['data_dir_prefix']
		else:
			data_dir_prefix = ''
		if 'class_names' in app.logger.results['platform']:
			cls_names = app.logger.results['platform']['class_names']
		else:
			cls_names = ['Con', 'SPC', 'PE', 'AT']
		#dcm_scan_param_csv_file = '/home/pj-019468-si/BARDA_ID/Code/barda_lus/blines/image_proc_algorithm/Read_params_from_DICOMS/output/medstar_dicom_scan_params_updated.csv'
		dcm_scan_param_csv_file = '/home/pj-019468-si/radhika/DICOM_details_Medstar.csv'

		if os.path.exists(data_dir_prefix + video_name + '.npz') or os.path.exists(video_name + '.npz'):
			painted_imgs = get_ori_video(video_name, app.logger, box_display_method=box_display_method, box_display_method_variable=box_display_method_variable,
										 img_type='npz', cls_names=cls_names)
		elif os.path.exists(data_dir_prefix + video_name + '.dcm.cropped.npz') or os.path.exists(video_name + '.dcm.cropped.npz'):
			painted_imgs = get_ori_video(video_name, app.logger, box_display_method=box_display_method, box_display_method_variable=box_display_method_variable,
										 img_type='dcm_npz', cls_names=cls_names)
		else:
			painted_imgs = get_ori_video(video_name, app.logger, box_display_method=box_display_method, box_display_method_variable=box_display_method_variable,
										 img_type='jpg', cls_names=cls_names,
										 dcm_scan_param_csv_file=dcm_scan_param_csv_file)
		img_src_dict = cache_display_images(painted_imgs,video_name,frame_names,cache_dir,dir_path,box_display_method,box_display_method_variable)
	return {'img_src_dict': img_src_dict}


@app.route("/clear_cache")
def clear_cache():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	cache_dir = 'static/cache_imgs'
	if os.path.exists(dir_path+'/'+cache_dir):
		shutil.rmtree(dir_path+'/'+cache_dir)
	return {'success': 'success'}