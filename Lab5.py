import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import *
from concurrent.futures import ThreadPoolExecutor
import time

n = 3
n_threads = 4

video = cv2.VideoCapture('17sec-compressed.mp4')

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

buf1 = np.empty((frames_count, frame_height, frame_width, 3),
                np.dtype('uint8'))
buf2 = np.empty((frames_count, frame_height, frame_width, 3),
                np.dtype('uint8'))
frame_i = 0
ret = True
print(frame_width, frame_height, frames_count, fps)

while (frame_i < frames_count and ret):
	ret, buf1[frame_i] = video.read()
	frame_i += 1
buf2 = buf1.copy()


class Changed_pixel:
	def __init__(self, row, col, distance):
		self.row = row
		self.col = col
		self.distance = distance

	def check_border(self, map, border_list):
		row_map = map[self.row]
		if not (self.col + 1 in row_map and self.col - 1 in row_map):
			border_list.append(self)
		elif self.row + 1 in map and self.row - 1 in map:
			if not (self.col in map[self.row + 1] and self.col in map[self.row - 1]):
				border_list.append(self)
		else:
			border_list.append(self)

	def __gt__(self, other):
		return self.distance > other.distance

	def __lt__(self, other):
		return self.distance < other.distance

	def __eq__(self, other):
		return self.distance == other.distance

	def __ge__(self, other):
		return self.distance >= other.distance

	def __le__(self, other):
		return self.distance <= other.distance


def draw_pixel(pixel):
	pixel[0] = 0
	pixel[1] = 0
	pixel[2] = 255


def draw_pixel2(pixel):
	pixel[0] = 0
	pixel[1] = 255
	pixel[2] = 255


def draw_fat_pixel(cur_target_frame, row, col):
	draw_pixel(cur_target_frame[row][col])
	draw_pixel(cur_target_frame[row + 1][col])
	draw_pixel(cur_target_frame[row - 1][col])
	draw_pixel(cur_target_frame[row][col + 1])
	draw_pixel(cur_target_frame[row][col - 1])


# ALG
def one_frame_processing(frame_i):
	print(frame_i)
	cur_frame = buf1[frame_i].copy()
	mapped_frame = buf1[frame_i - n].copy()
	cur_target_frame = buf2[frame_i]
	sub_mtx = np.subtract(cur_frame, mapped_frame, dtype = np.dtype('int16'))
	changed_pixels = []
	for str_i in range(frame_height):
		for col_i in range(frame_width):
			# cur_target_pixel = cur_target_frame[str_i][col_i]
			cur_sub = sub_mtx[str_i][col_i]
			# square = np.square(cur_frame[str_i][col_i] - mapped_frame[str_i][col_i])
			# sum_square = np.sum(square)
			# distance = np.sqrt(sum_square)
			# distance = np.sum(cur_frame[str_i][col_i] - mapped_frame[str_i][col_i])
			# distance = np.sqrt(np.sum(np.square(cur_sub)))
			if cur_sub[0] >= 10 or cur_sub[1] >= 10 or cur_sub[2] >= 10:
				# cur_target_pixel[0] = 0
				# cur_target_pixel[1] = 0
        		# cur_target_pixel[2] = 255
        		changed_pixels.append(Changed_pixel(str_i, col_i, 
					np.sqrt(np.sum(np.square(cur_sub)))))
	changed_pixels.sort(reverse=True)
	changed_pixels = changed_pixels[:len(changed_pixels)//10]
	pixels_map = {}
	for pixel in changed_pixels:
		if not(pixel.row in pixels_map):
			pixels_map[pixel.row] = {}
		pixels_map[pixel.row][pixel.col] = pixel
	border_list = []
	for key_row in pixels_map:
    	row_map = pixels_map[key_row]
		for key_col in row_map:
			row_map[key_col].check_border(pixels_map, border_list)
	# for pixel in changed_pixels:
	#  pixel_col = pixel.col
	#  pixel_row = pixel.row
	#  draw_pixel2(cur_target_frame[pixel_row][pixel_col])
  
	for pixel in border_list:
		pixel_col = pixel.col
		pixel_row = pixel.row
		if pixel_row < frame_height-1 and pixel_row > 0 and pixel_col < frame_width-1 and pixel_col > 0:
			draw_fat_pixel(cur_target_frame, pixel.row, pixel.col)
		else:
			draw_pixel(cur_target_frame[pixel_row][pixel_col])
# ALG\

arr = np.arange(n, frames_count)
start = time.time()
# Create a ThreadPoolExecutor with the specified number of threads 
with ThreadPoolExecutor(max_workers=n_threads) as executor: 
	# Use the executor to map the function to the array in parallel 
	executor.map(one_frame_processing, arr)
finish = time.time()
print(finish - start)

size = frame_width, frame_height
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), isColor = 3)
for i in range(frames_count):
	data = buf2[i]
	out.write(data)
out.release()
