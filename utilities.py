import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt

def toGrayScale(img):
	#return rgb2gray(img).astype(np.uint8)
	return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
	return img[::2, ::2]

def preprocess(img, FRAME_WIDTH, FRAME_HEIGHT):
	img = np.uint8(resize(rgb2gray(img), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
	#img = toGrayScale(img)
	#img = resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
	return img

def preprocess2(img, FRAME_WIDTH, FRAME_HEIGHT):
	processed_img = np.uint8(resize(rgb2gray(img),(FRAME_WIDTH, FRAME_HEIGHT))*255)
	print ('after shape:{}'.format(processed_img.shape))
	return np.reshape(processed_img, (1, FRAME_WIDTH, FRAME_HEIGHT))

def lambda_out_shape(input_shape):
	shape = list(input_shape)
	shape[-1] = 1
	return tuple(shape)

def list2np(in_list):
	return np.float32(np.array(in_list))

def transform_reward(reward):
	return np.sign(reward)

def plot_cost(cost_history):
	matplotlib.use("MacOSX")
	plt.plot(np.arange(len(cost_history)), cost_history)
	plt.ylabel('Cost')
	plt.xlabel('Training Steps')
	plt.show()