# Predicting Big Five personality traits using facial images

This repository contains code to pre-preprocess data from the [Chalearn First Impressions V2 Challenge](https://chalearnlap.cvc.uab.cat/dataset/24/description/), and subsequently, train various PyTorch models to predict Big Five scores on facial images. In principle, it can be used with other datasets too, which contain videos of human faces and their respective Big Five scores.

## The dataset
The First Impressions dataset consists of 10.000 video clips of around 3000 high-quality YouTube videos, which are split into train, validation and test with a 3:1:1 ratio. The labels (Big Five scores) have been crowdsourced for every video using Amazon Mechanical Turk. For more information, visit the Challenge's [Website](https://chalearnlap.cvc.uab.cat/dataset/24/description/). 

The below image was given by the challenge authors [[source](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_32)] and nicely illustrates the dataset.
![Illustration of dataset](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-319-49409-8_32/MediaObjects/435559_1_En_32_Fig3_HTML.gif)

## Data pre-processing
To train a predictive model, one can focus on facial images, as papers have shown that the audio sequence of videos does not yield substantial improvements for personality trait predictions. Accordingly, one has to extract face image chips from videos first. 

The pre-processing has two steps:
1. Extract N frames at random positions from each video clip
```
python pre-processing/extract_frames_from_videos.py path_to_video_clips path_to_frame_directory N
```
Hereby, we should choose N=1 for the validation and test dataset. For the training dataset, we may choose N>1, which effectively acts as data augmentation as we're drawing different facial expressions from a single video. For the results reported in this repo, N=30 was chosen, but the winning team of the original competition reported using N=92 [[paper](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_25)].

2. Detect faces and crop 224px x 224px images for training
```
python pre-processing/extract_faces_from_frames.py path_to_frame_directory path_to_face_directory
```
This code uses a pre-trained facial recognition model from the [Dlib](http://dlib.net/) library. The final images used for training, validation and testing will reside in `path_to_face_directory`.
