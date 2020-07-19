import cv2
import os
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_folder", type=str, required=True,
				help="folder to clean similar images")
ap.add_argument("-o", "--output_folder", type=str, required=True,
				help="folder to move similar images")
ap.add_argument("-s", "--hash_size", type=int, default=32,
				help="hash size")
ap.add_argument("-t", "--threshold", type=int, required=True,
				help="threshold for detecting similar images")

args = vars(ap.parse_args())

def calculate_mean(pixels_list):
	sum = 0
	mean = 0

	for i in range(len(pixels_list)):
		sum = sum + pixels_list[i]

	mean = sum / len(pixels_list)

	return mean

def grab_pixels(squeezed_frame):
	pixels_list = []

	for x in range(0, squeezed_frame.shape[1], 1):
		for y in range(0, squeezed_frame.shape[0], 1):
			pixel_color = squeezed_frame[x, y]
			pixels_list.append(pixel_color)

	return pixels_list

def make_bits_list(mean, pixels_list):
	bits_list = []

	for i in range(len(pixels_list)):
		if (pixels_list[i] >= mean):
			bits_list.append(255)
		else:
			 bits_list.append(0)

	return bits_list

def hashify(squeezed_frame, bits_list):
	bitIndex = 0
	hashed_frame = squeezed_frame

	for x in range(0, squeezed_frame.shape[1], 1):
			for y in range(0, squeezed_frame.shape[0], 1):
				hashed_frame[x, y] = bits_list[bitIndex]
				bitIndex += 1

	return hashed_frame

def hash_generator_animation(frame, hash_size, iterations):
	fourcc = cv2.VideoWriter_fourcc(* "MJPG")
	writer = cv2.VideoWriter(f"static/test.avi", fourcc, 25, (frame.shape[1]*2, frame.shape[0]), True)

	for i in range(iterations):
		if (hash_size >= 16):
			frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
			frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
			pixels_list = grab_pixels(frame_squeezed)
			mean_color = calculate_mean(pixels_list)
			bits_list = make_bits_list(mean_color, pixels_list)
			hashed_frame = hashify(frame_squeezed, bits_list)
			hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
			#hashed_frame = cv2.resize(hashed_frame, (frame.shape[1], frame.shape[0]))
			cv2.putText(hashed_frame, f"hash_size: {hash_size}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, lineType=cv2.LINE_AA)

			im_v = cv2.hconcat([frame, hashed_frame])
			cv2.imshow("dfs", im_v)
			cv2.waitKey(1)
			writer.write(im_v)
			hash_size -= 1

def generate_hash(frame, hash_size):
	frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
	frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
	pixels_list = grab_pixels(frame_squeezed)
	mean_color = calculate_mean(pixels_list)
	bits_list = make_bits_list(mean_color, pixels_list)
	hashed_frame = hashify(frame_squeezed, bits_list)
	hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
	#hashed_frame = cv2.resize(hashed_frame, (128, 128))

	return bits_list, hashed_frame

def clean_folder(input_folder, output_folder, hash_size, threshold):
	files = (os.listdir(input_folder))
	list_length = len(files)
	sum_diff = 0
	i = 0
	k = 1
	frame = None
	hashed_frame = None
	dublicate_count = 0

	while (i < len(files)):
		sum_diff = 0

		if (files[i] != None):
			#frame = cv2.imread(f"{input_folder}/person3763.jpg")
			frame = cv2.imread(f"{input_folder}/{files[i]}")
			#frame = cv2.GaussianBlur(frame, (13,13),13)
			bits_list, hashed_frame = generate_hash(frame, hash_size)

		while (k < len(files)):
			if (i != k) & (files[k] != None):
				new_frame = cv2.imread(f"{input_folder}/{files[k]}")
				#new_frame = cv2.GaussianBlur(new_frame, (13,13),13)
				newbits_list, hashed_second_frame = generate_hash(new_frame, hash_size)

				for j in range(len(bits_list)):
					if (bits_list[j] != newbits_list[j]):
						sum_diff += 1

				print(f"{files[i]} -> {files[k]} sum_diff = {sum_diff}")

				im_h = cv2.hconcat([cv2.resize(frame, (450, 450)), cv2.resize(new_frame, (450, 450))])
				im_h2 = cv2.hconcat([cv2.resize(hashed_frame, (450, 450)), cv2.resize(hashed_second_frame, (450, 450))])
				im_v = cv2.vconcat([im_h, im_h2])

				if (sum_diff < threshold):
					Path(f"{input_folder}/{files[k]}").rename(f"{output_folder}/{files[k]}")
					print(f"Deleted {k} element ({files[k]}) of {list_length}")
					del files[k]
					dublicate_count += 1
					cv2.putText(im_v, f"FOUND COPY", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, lineType=cv2.LINE_AA)					
				else:					
					k += 1

				cv2.putText(im_v, f"SIMILAR: {dublicate_count}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, lineType=cv2.LINE_AA)
				cv2.imshow("hasher", im_v)
				cv2.waitKey(1)						

				sum_diff = 0
		i += 1
		k = i + 1

clean_folder(args['input_folder'], args['output_folder'], args['hash_size'], args['threshold'])

#input_frame = cv2.imread(f"images/harold.jpg")
#input_frame = cv2.imread(f"images/car/car1066.jpg")

#bits_list, outputHash = generate_hash(input_frame, 16)

# cv2.imshow("hash", outputHash)
# cv2.waitKey(0)
#hash_generator_animation(input_frame, 512, 512)

#Example: python image_hash.py -i images/all_images/ -o images/s -s 16 -t 60