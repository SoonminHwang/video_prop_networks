import cv2
import glob
import os
import numpy as np
# from skvideo.io import VideoWriter

def make_video(result_path, davis_path, save_path, fps=5, size=None, vid_format="MJPG"):

	fourcc = cv2.VideoWriter_fourcc(*vid_format)
	vid = None
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	sequences = os.listdir(result_path)

	for seq in sequences:	
		print('### Sequence: %s' % seq)

		image_path = os.path.join(davis_path, 'JPEGImages', '480p', seq)
		gt_path = os.path.join(davis_path, 'Annotations', '480p', seq)

		pred_names = glob.glob(os.path.join(result_path, seq, '*.png'))
		pred_names.sort()
				
		for pred_name in pred_names:

			name = os.path.basename(pred_name)

			pred = cv2.imread( pred_name )
			gt = cv2.imread( os.path.join(gt_path, name) )
			img = cv2.imread( os.path.join(image_path, name.replace('png', 'jpg')) )

			cropped = img * ( pred == 255 )
			
			analysis = np.zeros( (gt.shape[0], gt.shape[1]), np.uint8 )
			analysis[ (pred[:,:,0] == 255) * (gt[:,:,0] == 255) ] = 1		# TP (light grey)
			analysis[ (pred[:,:,0] == 255) * (gt[:,:,0] == 0) ] = 2			# FP (red)
			analysis[ (pred[:,:,0] == 0) * (gt[:,:,0] == 255) ] = 3			# FN (blue)
			analysis[ (pred[:,:,0] == 0) * (gt[:,:,0] == 0) ] = 4			# TN (dark grey)

			palette = np.array( [[0, 0, 0], [192, 192, 192], [128, 0, 0], [0, 0, 128], [64, 64, 64]], np.uint8)

			color_image = palette[analysis.ravel()].reshape(img.shape)        
			color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

			dummy = np.zeros( img.shape, np.uint8 )



			
			# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_TRIPLEX, 1, 1, 0, 3, 8) #Creates a font
			# text_opt = dict(org=(10,10), fontFace=3, fontScale=1, 
			# 	color=(255,255,255), thickness=2, font=cv2.FONT_HERSHEY_TRIPLEX)

			font = cv2.FONT_HERSHEY_DUPLEX

			gt = cv2.putText(np.copy(gt), "GT", (30,50), font, 1, (255,255,255))
			pred = cv2.putText(np.copy(pred), "Prediction", (30,50), font, 1, (255,255,255))
			cropped = cv2.putText(np.copy(cropped), "Cropped", (30,50), font, 1, (255,255,255))
			# color_image = cv2.putText(np.copy(color_image), "TP(light grey), FP(red), FN(blue), TN(dark grey)", (10,50), font, 0.5, (255,255,255))
			# color_image = cv2.putText(np.copy(color_image), "TP(light grey)", (30,50), font, 1, (255,255,255))
			color_image = cv2.putText(np.copy(color_image), "FP(red)", (30,50), font, 1, (255,255,255))
			color_image = cv2.putText(np.copy(color_image), "FN(blue)", (30,90), font, 1, (255,255,255))
			# color_image = cv2.putText(np.copy(color_image), "TN(dark grey)", (30,170), font, 1, (255,255,255))



			# gt = cv2.putText(img=np.copy(gt), text="GT", **text_opt)
			# pred = cv2.putText(img=np.copy(pred), text="Prediction", **text_opt)
			# cropped = cv2.putText(img=np.copy(cropped), text="Cropped", **text_opt)

			if vid is None:				
				outvid = os.path.join(save_path, seq + '.avi')
				if size is None:
					size = img.shape[1], img.shape[0]
				vid = cv2.VideoWriter(outvid, fourcc, float(fps), (3*img.shape[1], 2*img.shape[0]))
				# vid = VideoWriter(outvid, frameSize=size)
				# vid.open()

			if size[0] != img.shape[1] and size[1] != img.shape[0]:
				img = resize(img, size)
				pred = resize(pred, size)
				gt = resize(gt, size)
				cropped = resize(cropped, size)
				color_image = resize(color_image, size)
				dummy = resize(dummy, size)

			frame = np.vstack( (np.hstack((img, pred, gt)), np.hstack((cropped, color_image, dummy))) )				
			
			vid.write(frame)

		vid.release()
		vid = None


if __name__ == '__main__':

	result_path = '/home/rcvlab/workspace/video_segmentation/vps-caffe/data/seg_results/STAGE1_RESULT/'
	davis_path = '/home/rcvlab/workspace/video_segmentation/vps-caffe/data/DAVIS/'
	save_path = '/home/rcvlab/workspace/video_segmentation/vps-caffe/videos/'
	make_video( result_path, davis_path, save_path )