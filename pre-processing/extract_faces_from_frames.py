"""This script iterates through a given directory containing only JPG images, runs a facial detection model on every image, and if successful, crops and stores 224px x 224px image chips around those faces in the target directory."""
import sys
import os
import random
import time
import shutil
from multiprocessing import Pool

from PIL import Image, ImageStat
import pandas as pd
import tqdm
import dlib


if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./extract_faces_from_frames.py frame_directory face_target_directory path_to_face_detection_model \n"
    )
    exit()

frame_dir = str(sys.argv[1])
face_target_dir = str(sys.argv[2])
predictor_path = str(sys.argv[3])

# Get list of frames to be processed
all_frames = os.scandir(frame_dir)
all_frames = [frame.name for frame in all_frames]
print(len(all_frames), 'files found in all_frames')

# Check which frames have already been processed and exist in target_dir
existing_frames = os.scandir(face_target_dir)
existing_frames = [frame.name.split('___')[-1].replace('_aligned', '') for frame in existing_frames]
existing_frames = set(existing_frames)
new_frames = [frame_dir + frame for frame in tqdm.tqdm(all_frames) if frame not in existing_frames]
print(len(new_frames), 'new files to be processed.')


def extract_face(face_file_path):
    """Try to detect a face in JPG file with face_file_path, horizontally align that image and save a 224px x 224px cropped around the face to face_target_dir. 
    
    The corresponding images are saved to the target directory like <image_name>_aligned.jpg. face_target_dir (int) should be defined globally.
    
    Args:
        face_file_path (str): Path to a JPG file.
    """
    image_path = face_file_path.split('/')[-1]
    # Load models
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    
    # Load the image
    img = dlib.load_rgb_image(face_file_path)
    
    # Ask the detector to find the bounding boxes of each face.
    dets = detector(img, 1)
    
    try:
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(face_file_path))
            return

        # Find landmarks
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        # Get the aligned face images
        images = dlib.get_face_chips(img, faces, size=224)
        for image in images:
            # Compute confidence score as well
            dets, scores, idx = detector.run(image, 1, -1)
            aligned_image = Image.fromarray(image) 
            file_name = face_target_dir + image_path[:-4] + '_aligned.jpg'
            aligned_image.save(file_name, "JPEG") 
    except:
        print(image_path, 'resulted in some error')

print('Extracting faces now.')    
start = time.time()
p = Pool(24)
for _ in tqdm.tqdm(p.imap_unordered(extract_face, new_frames), total=len(new_frames)):
    pass
# p.map(extract_face, file_list)
run_time = time.time() - start
print("Took {} seconds per image.".format(run_time / len(new_frames)))
