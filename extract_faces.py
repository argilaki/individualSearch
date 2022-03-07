import cv2
from PIL import Image
import os
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np




def extract_frames(video_path, video_name, sceneChangeRatio=0.35):

	#make a directory for the extracted frames
	os.system("mkdir %s/%s" %(video_path,video_name))

	command = ("ffmpeg -i ./%s/%s.mp4 -q:v 2 -vf select='gt(scene\,%s)' -s 720x720 -vsync 0 ./%s/%s/frame%%d.jpg" 
		%(video_path, video_name, str(sceneChangeRatio), video_path, video_name))
	# command = ("ffmpeg -i ./videos/%s.mp4 -q:v 2 -vf select='eq(pict_type\,PICT_TYPE_I)' -s 720x720 -vsync 0 ./videos/%s/frame%%d.jpg" %(vid,vid))
	os.system(command)
	print('All frames was extracted from video successfully...')



def detect_face(video_path, video_name, required_size=(160, 160)):
	# create the detector, using default weights
	detector = MTCNN()
	os.system("mkdir %s/%s_faces" %(video_path,video_name))

	# now we search for faces in frames that is already extracted
	frames = os.listdir('%s/%s'%(video_path, video_name))
	count = 0

	for items in frames:
		img = Image.open('%s/%s/'%(video_path,video_name) + items)

		# # # convert to array
		pixels = asarray(img)
		# detect faces in the image
		results = detector.detect_faces(pixels)
		# if no faces were detected.
		if not results:
			pass
		# if one or more faces were detected.
		elif len(results) >= 1:
			for result in results:
				if result['confidence'] >= 0.95:
					# extract the bounding box from the first face
					x1, y1, width, height = result['box']
					# bug fix
					x1, y1 = abs(x1), abs(y1)
					x2, y2 = x1 + width, y1 + height
					# extract the face
					face = pixels[y1-5:y2+5, x1-5:x2+5]
					# resize pixels to the model size
					image = Image.fromarray(face)
					image = image.resize(required_size)
					face_array = asarray(image)
					face_array = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
					cv2.imwrite("%s/%s_faces/face%d.jpg"%(video_path, video_name, count), face_array)
					count = count + 1
	print('%s faces detected in the %s file' %(str(count), video_name))
	return count				





if __name__ == '__main__':

	path = './videos'
	vid = 'input'
	sceneChangeRatio = 0.35

#	extract_frames(path, vid, sceneChangeRatio)
	detect_face(path, vid)