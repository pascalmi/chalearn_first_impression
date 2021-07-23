# Predicting Big Five personality traits using facial images

This lightweight repository contains code to pre-preprocess data from the [Chalearn First Impressions V2 Challenge](https://chalearnlap.cvc.uab.cat/dataset/24/description/), and subsequently, train various PyTorch models to predict Big Five scores on facial images. In principle, it can be used with other datasets too, which contain videos of human faces and their respective Big Five scores.

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

## Training and Hyperparameteroptimization
In this repo, hyperparameteroptimization is taken care of automatically by [Raytune](https://docs.ray.io/en/master/index.html). To configure the desired Hyperparameter search space, you can simply edit the config variable in train_resnet_raytune_hyperparams.py. Now, we train models via
```
python train_resnet_raytune_hyperparams.py num_trials num_epochs experiment_name model_name
```
where `num_trials` is the maximum number of Hyperparameter configurations, `num_epochs` the maximum number of epochs per trial (however, depending on your computational resources, you also want to keep a close eye on Raytune's grace_period), `experiment_name` the name of the directory where training results will be logged via Tensorboard, and `model_name` picks a model from `model_architectures.py`. 

Once the ideal Hyperparameter configuration was found, we can run a final training, which also incorporates the validation dataset for training, via 
```
python train.py num_epochs experiment_name model_name true batch_size
```

## Results
The following table compares "accuracies" (see [paper](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_25) for their derivation) for the different big five scores and their mean towards the winning team (NJU-LAMDA) of the original Chalearn Challenge.
| Metrics | NJU-LAMDA | This Repo | 
| ------- | :----------------------------------------: | :---------: | 
| Acc. Open. | 0.9123 | 0.8984 | 
| Acc. Consc. | 0.9166 | 0.8950 | 
| Acc. Extra. | 0.9133 | 0.8965 | 
| Acc. Agree. | 0.9126 | 0.9028 | 
| Acc. Neuro. | 0.9100 | 0.8933 | 
| Mean Acc. | 0.9130 | 0.8972 | 

As we see, this lightweight repo achieves results close to the state-of-the-art and that although it does not incorporate the audio sequence of the videos, as did team NJU-LAMDA. 

For a more comprehensive visualization of the results, the following plot shows the distribution of ground truth and prediction scores exemplarily for Extraversion, including the resulting scatter plot and error term distribution which all show a clear and strong correlation between predictions and ground truth values and as we can see, we're significantly outperforming the random baseline (red line, right plot).
![results plot](https://github.com/pascalmi/chalearn_first_impression/blob/main/media/test_evaluation.png)

