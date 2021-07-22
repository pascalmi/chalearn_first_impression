"""This script iterates through a given video directory, extracts N random frames from every video file in that directory and saves those frames as JPG to a given target directory. The script assumes that the given video directory contains only .mp4 files."""

import os
import sys
import time
import random
from multiprocessing import Pool

from PIL import Image
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import *
import tqdm

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./extract_frames_from_videos.py video_directory target_directory num_frames_per_video \n"
    )
    exit()

video_dir = str(sys.argv[1])
frame_target_dir = str(sys.argv[2])
num_frames_per_video = int(sys.argv[3])

# Get a list of all video files
video_list = os.listdir(video_dir)
video_list = [video_dir + file for file in video_list]

# Get a list of existing frames in the target directory (in case we need to re-run the algorithm)
frame_list = os.scandir(frame_target_dir)
frame_list = [frame.name for frame in frame_list]
existing_videos = [frame.split('__')[0] for frame in frame_list]
existing_videos = list(set(existing_videos))
print(len(existing_videos), 'videos were already extracted to the target directory.')

# Remove videos that already exist in target directory from our process
video_list = [video for video in video_list if not video.split('/')[-1].split('__')[0] in existing_videos]
print(len(video_list), 'new videos to be processed.')

def extract_frames(video):
    """Extracts N images (frames) at random times from a given .mp4 file "<video_name>.mp4".
    
    The corresponding images are saved to the target directory like <video_name>__frame_i.jpg, where 
    i is the frame number of the original video. num_frames_per_video (int) should be globally defined.
    
    Args:
        video (str): Path to the video file.
    """
    try:
        video_file_name = video.split('/')[-1][:-4]
        clip = VideoFileClip(video)
        total_num_frames = clip.reader.nframes
        total_frames = list(range(1,total_num_frames))
        random.shuffle(total_frames)
        for i in total_frames[0:num_frames_per_video]:
            numpy_im = clip.get_frame(i)
            pil_im = Image.fromarray(numpy_im)
            pil_im.save(frame_target_dir + video_file_name + '__frame_' + str(i) + '.jpg')
        return 1
    except:
        print("{} resulted in an error.".format(video))
        return 0
    

start = time.time()
p = Pool(24) # Use 24 threads in parallel
for _ in tqdm.tqdm(p.imap_unordered(extract_frames, video_list), total=len(video_list)):
     pass
run_time = time.time() - start
print('Frame extraction took {} seconds per video.'.format(run_time / len(video_list)))
    