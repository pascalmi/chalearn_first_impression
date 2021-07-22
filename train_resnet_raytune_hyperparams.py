import os
import time
import sys
import random
import pickle
import math
import itertools
from functools import partial

import pandas as pd
import numpy as np
import skimage
from skimage import io, transform
import matplotlib.pyplot as plt
from ipywidgets import IntProgress
from scipy.stats import spearmanr, pearsonr

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from model_architectures import load_model # load_model defines and loads different ResNet architectures


# Helper function to format the label files such that filenames are primary key values of a dictionary
# that contains the corresponding big five scores for each file
def format_label_list(label_dict):
    # IMPORTANT: LABEL ORDER IS 'O-C-E-A-N'
    label_keys_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    new_label_dict = {}
    for key in label_keys_order:
        # Remove unused 'interview' label from the .pkl files given by Chalearn
        if key == 'interview':
            continue;
        for sample in label_dict[key]:
            if sample not in new_label_dict:
                new_label_dict[sample] = [label_dict[key][sample]]
            else:
                new_label_dict[sample].append(label_dict[key][sample])
    return new_label_dict


class ChalearnDataset(Dataset):
    """ChaLearn First Impression Dataset"""
    
    def __init__(self, label_dict, root_dir, transform=None):
        """
        Args:
            label_dict (dictionary): Dictionary with (original) filenames as keys and big 5
                label dictionaries {extraversion: 0.3842833, neuroticism: 0.742332} as values.
            root_dir (string): Directory with all the images.
                Original filename can be retrieved as
                label_key = img_name[:15] + '.mp4'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = label_dict
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.image_names[idx]
        
        # Load image
        img_path = self.root_dir + img_name
        image = io.imread(img_path) / 255
        image = torch.Tensor(image).float()
        image = image.reshape(3, 224, 224)
        
        # Get the labels
        label_key = img_name[:15] + '.mp4'
        big_five_scores = self.labels[label_key]
        big_five_scores = torch.FloatTensor(big_five_scores)
        
        if self.transform:
            image = self.transform(image)
        
        return image, big_five_scores


def get_test_statistics(testset, model, device="cpu", best_dir=None):
    best_dir = '/'.join(best_dir.split('/')[:-2]) + '/'
    testloader = torch.utils.data.DataLoader(
        testset,batch_size=4, shuffle=False, num_workers=2)
    model.eval()
    
    test_scores = torch.empty(0,5)
    test_labels = torch.empty(0,5)
    test_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.reshape(inputs.shape[0], 3, 224, 224)
            inputs, labels = inputs.to(device), labels.to(device)

            ps = model.forward(inputs)
            loss = criterion(ps, labels)
            test_loss += loss.item()

            ps_cpu = ps.cpu()
            labels_cpu = labels.cpu()

            test_scores = torch.cat((test_scores, ps_cpu))
            test_labels = torch.cat((test_labels, labels_cpu))
            
        # Compute pearson correlations
        pearson_test_correlation_O, _ = pearsonr(test_scores[:,0].tolist(), test_labels[:,0].tolist())
        pearson_test_correlation_C, _ = pearsonr(test_scores[:,1].tolist(), test_labels[:,1].tolist())
        pearson_test_correlation_E, _ = pearsonr(test_scores[:,2].tolist(), test_labels[:,2].tolist())
        pearson_test_correlation_A, _ = pearsonr(test_scores[:,3].tolist(), test_labels[:,3].tolist())
        pearson_test_correlation_N, _ = pearsonr(test_scores[:,4].tolist(), test_labels[:,4].tolist())
        accuracy_O = compute_label_accuracy(test_labels[:,0].tolist(), test_scores[:,0].tolist())
        accuracy_C = compute_label_accuracy(test_labels[:,1].tolist(), test_scores[:,1].tolist())
        accuracy_E = compute_label_accuracy(test_labels[:,2].tolist(), test_scores[:,2].tolist())
        accuracy_A = compute_label_accuracy(test_labels[:,3].tolist(), test_scores[:,3].tolist())
        accuracy_N = compute_label_accuracy(test_labels[:,4].tolist(), test_scores[:,4].tolist())
        total_mean_accuracy = (accuracy_O + accuracy_C + accuracy_E + accuracy_A + accuracy_N) / 5

    if best_dir:
        eval_df = pd.DataFrame(data={'Labels':test_labels[:,0].tolist(), 'Predictions':test_scores[:,0].tolist()})
        eval_df.to_csv(best_dir + 'test_results.csv', index=False)
    
    return test_loss / len(testloader), pearson_test_correlation_O, pearson_test_correlation_C, pearson_test_correlation_E, pearson_test_correlation_A, pearson_test_correlation_N, accuracy_O, accuracy_C, accuracy_E, accuracy_A, accuracy_N, total_mean_accuracy

def compute_label_accuracy(list_ground_truth, list_predictions):
    if len(list_ground_truth) != len(list_predictions):
        print("Cannot compute label accuracy, list lengths do not match!")
        exit()
    num_predictions = len(list_predictions)
    one_minus_absolute_distances = [1 - abs(list_ground_truth[i] - list_predictions[i]) for i in range(num_predictions)]
    return sum(one_minus_absolute_distances) / num_predictions
        


def train_model(config, model_name, trainset, valset, max_num_epochs, checkpoint_dir=None):
    
    model = load_model(model_name, config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(net)
    model.to(device)
    
    params = list(model.fc.parameters())
    # layer1_params = list(model.layer1.parameters())
    # layer2_params = list(model.layer2.parameters())
    # layer3_params = list(model.layer3.parameters())
    # layer4_params = list(model.layer4.parameters())
    criterion = nn.MSELoss()
    
    # Use this if: Re-training all 4 layers of our ResNet
    """
    optimizer = optim.Adam([{'params': params},
                            {'params': layer1_params, 'lr': 10e-7},
                            {'params': layer2_params, 'lr': 10e-7},
                            {'params': layer3_params, 'lr': 10e-5},
                            {'params': layer4_params, 'lr': 10e-5}],
                            lr=config["lr"],
                            weight_decay=config["weight_decay"])
    """
    # Use this if: Only training the FC layer
    optimizer = optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    
    # In case we want to resume a training at a specific checkpoint
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(config["batch_size"]), num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, num_workers=8)
    
    # Initialize some variables for training
    steps = 0
    print_every = len(trainloader) # Evaluate model at the end of every epoch only
    train_loss = 0
    train_losses, val_losses, train_correlations, val_correlations = [], [], [], []
    
    # Initialize variables and files for model saving
    last_epoch_O = 0
    last_epoch_C = 0
    last_epoch_E = 0
    last_epoch_A = 0
    last_epoch_N = 0
    
    last_steps_O = 0
    last_steps_C = 0
    last_steps_E = 0
    last_steps_A = 0
    last_steps_N = 0
    
    last_pearson_O = 0
    last_pearson_C = 0
    last_pearson_E = 0
    last_pearson_A = 0
    last_pearson_N = 0
    
    with tune.checkpoint_dir(0) as checkpoint_dir:
        trial_dir = '/'.join(checkpoint_dir.split('/')[:-1])
        torch.save(model,trial_dir + '/' + 'O' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_pearson_O)}")
        torch.save(model,trial_dir + '/' + 'C' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_pearson_C)}")
        torch.save(model,trial_dir + '/' + 'E' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_pearson_E)}")
        torch.save(model,trial_dir + '/' + 'A' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_pearson_A)}")
        torch.save(model,trial_dir + '/' + 'N' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_pearson_N)}")
    
    # Helper function to save the best model per trait
    def save_model_per_trait(last_correlation, last_steps, last_epoch, correlation, steps, epoch, model_path, model, trait):
        max_correlation = correlation
        os.remove(model_path + '/' + trait + '_' + f"epoch{last_epoch}_step{last_steps}_corr{'%.4f'%(last_correlation)}")
        torch.save(model,model_path + '/' + trait + '_' + f"epoch{epoch}_step{steps}_corr{'%.4f'%(correlation)}")
        return correlation, steps, epoch
    
    
    print("Starting to train now")
    # Start Training loop
    for epoch in range(max_num_epochs):
    # Append results to those tensors to calculate correlations
        train_scores = torch.empty(0,5)
        val_scores = torch.empty(0,5)
        train_labels = torch.empty(0,5)
        val_labels = torch.empty(0,5)

        # Train the model
        for inputs, labels in trainloader:
            steps += 1
            inputs = inputs.reshape(inputs.shape[0], 3, 224, 224)
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            ps = model.forward(inputs)
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            ps_cpu = ps.cpu()
            labels_cpu = labels.cpu()
            train_scores = torch.cat((train_scores, ps_cpu))
            train_labels = torch.cat((train_labels, labels_cpu))

            # Evaluate the model
            if steps % print_every == 0:
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs = inputs.reshape(inputs.shape[0], 3, 224, 224)
                        inputs, labels = inputs.to(device), labels.to(device)

                        ps = model.forward(inputs)
                        loss = criterion(ps, labels)
                        val_loss += loss.item()

                        ps_cpu = ps.cpu()
                        labels_cpu = labels.cpu()
                        val_scores = torch.cat((val_scores, ps_cpu))
                        val_labels = torch.cat((val_labels, labels_cpu))

                    # Calculate Statistics
                    # Remember: Labels are ordered like 'O-C-E-A-N'
                    
                    # Opennness 
                    pearson_train_correlation_O, _ = pearsonr(train_scores[:,0].tolist(), train_labels[:,0].tolist())
                    pearson_val_correlation_O, _ = pearsonr(val_scores[:,0].tolist(), val_labels[:,0].tolist())
                    accuracy_O = compute_label_accuracy(val_labels[:,0].tolist(), val_scores[:,0].tolist())
                    # Conscientiousness
                    pearson_train_correlation_C, _ = pearsonr(train_scores[:,1].tolist(), train_labels[:,1].tolist())
                    pearson_val_correlation_C, _ = pearsonr(val_scores[:,1].tolist(), val_labels[:,1].tolist())
                    accuracy_C = compute_label_accuracy(val_labels[:,1].tolist(), val_scores[:,1].tolist())
                    # Extraversion
                    pearson_train_correlation_E, _ = pearsonr(train_scores[:,2].tolist(), train_labels[:,2].tolist())
                    pearson_val_correlation_E, _ = pearsonr(val_scores[:,2].tolist(), val_labels[:,2].tolist())
                    accuracy_E = compute_label_accuracy(val_labels[:,2].tolist(), val_scores[:,2].tolist())
                    # Agreeableness
                    pearson_train_correlation_A, _ = pearsonr(train_scores[:,3].tolist(), train_labels[:,3].tolist())
                    pearson_val_correlation_A, _ = pearsonr(val_scores[:,3].tolist(), val_labels[:,3].tolist())
                    accuracy_A = compute_label_accuracy(val_labels[:,3].tolist(), val_scores[:,3].tolist())
                    # Neuroticism
                    pearson_train_correlation_N, _ = pearsonr(train_scores[:,4].tolist(), train_labels[:,4].tolist())
                    pearson_val_correlation_N, _ = pearsonr(val_scores[:,4].tolist(), val_labels[:,4].tolist())
                    accuracy_N = compute_label_accuracy(val_labels[:,4].tolist(), val_scores[:,4].tolist())
                    
                    total_mean_accuracy = (accuracy_O + accuracy_C + accuracy_E + accuracy_A + accuracy_N) / 5

                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        trial_dir = '/'.join(checkpoint_dir.split('/')[:-1])
                        train_path = os.path.join(trial_dir, "train_results_epoch_" + str(epoch) + ".csv")
                        train_df = pd.DataFrame(data={'Labels':train_labels[:,0].tolist(),
                                                      'Predictions':train_scores[:,0].tolist()})
                        train_df.to_csv(train_path, index=False)
                        val_path = os.path.join(trial_dir, "val_results_epoch_" + str(epoch) + ".csv")
                        val_df = pd.DataFrame(data={'Labels':val_labels[:,0].tolist(), 
                                                    'Predictions':val_scores[:,0].tolist()})
                        val_df.to_csv(val_path, index=False)
                        # open(os.path.join(checkpoint_dir[:-13], 'epoch_' + str(epoch) + '_' + str(val_loss)), 'a').close()
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save((model.state_dict(), optimizer.state_dict()), path) # Attention: 44MB per trial per epoch!!
                    
                        ## Save models if new maximum validation correlation was reached for any trait
                        if pearson_val_correlation_O > last_pearson_O:
                            last_pearson_O, last_steps_O, last_epoch_O = save_model_per_trait(last_pearson_O, last_steps_O, last_epoch_O,
                                                                                             pearson_val_correlation_O, steps, epoch+1,
                                                                                             trial_dir, model, 'O')
                        if pearson_val_correlation_C > last_pearson_C:
                            last_pearson_C, last_steps_C, last_epoch_C = save_model_per_trait(last_pearson_C, last_steps_C, last_epoch_C,
                                                                                             pearson_val_correlation_C, steps, epoch+1,
                                                                                             trial_dir, model, 'C')
                        if pearson_val_correlation_E > last_pearson_E:
                            last_pearson_E, last_steps_E, last_epoch_E = save_model_per_trait(last_pearson_E, last_steps_E, last_epoch_E,
                                                                                             pearson_val_correlation_E, steps, epoch+1,
                                                                                             trial_dir, model, 'E')
                        if pearson_val_correlation_A > last_pearson_A:
                            last_pearson_A, last_steps_A, last_epoch_A = save_model_per_trait(last_pearson_A, last_steps_A, last_epoch_A,
                                                                                             pearson_val_correlation_A, steps, epoch+1,
                                                                                             trial_dir, model, 'A')
                        if pearson_val_correlation_N > last_pearson_N:
                            last_pearson_N, last_steps_N, last_epoch_N = save_model_per_trait(last_pearson_N, last_steps_N, last_epoch_N,
                                                                                             pearson_val_correlation_N, steps, epoch+1,
                                                                                             trial_dir, model, 'N')

                # Average loss across all minibatches
                train_loss = train_loss/print_every
                val_loss = val_loss/len(valloader)
              
                # Report results to Raytune
                tune.report(val_loss=val_loss, train_loss=train_loss,
                            train_correlation_O=pearson_train_correlation_O, val_correlation_O=pearson_val_correlation_O, 
                            train_correlation_C=pearson_train_correlation_C, val_correlation_C=pearson_val_correlation_C,
                            train_correlation_E=pearson_train_correlation_E, val_correlation_E=pearson_val_correlation_E,
                            train_correlation_A=pearson_train_correlation_A, val_correlation_A=pearson_val_correlation_A,
                            train_correlation_N=pearson_train_correlation_N, val_correlation_N=pearson_val_correlation_N,
                            accuracy_O=accuracy_O, accuracy_C=accuracy_C, 
                            accuracy_E=accuracy_E, accuracy_A=accuracy_A, accuracy_N=accuracy_N, 
                            total_mean_accuracy=total_mean_accuracy
                           )
            
                train_loss = 0
                model.train()
        
    print("Finished Trial.")
    writer.flush()


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1, experiment_name=None, model_name='ResNet18_v1.0'):
    
    # Set data related paths
    script_path = os.path.dirname(os.path.abspath( __file__ ))
    train_data_dir = script_path + '/' + 'data/train/faces/'
    valid_data_dir = script_path + '/' + 'data/valid/faces/'
    test_data_dir = script_path + '/' + 'data/test/faces/'
    train_labels_pickle = script_path + '/' + 'data/train/annotation_training.pkl'
    valid_labels_pickle = script_path + '/' + 'data/valid/annotation_validation.pkl'
    test_labels_pickle = script_path + '/' + 'data/test/annotation_test.pkl'

    # Load label pickle files
    train_annotations = pickle.load(open(train_labels_pickle, "rb" ), encoding="latin1" )
    val_annotations = pickle.load(open(valid_labels_pickle, "rb" ), encoding="latin1")
    test_annotations = pickle.load(open(test_labels_pickle, "rb" ), encoding="latin1")

    # Format labels 
    train_annotations = format_label_list(train_annotations)
    val_annotations = format_label_list(val_annotations)
    test_annotations = format_label_list(test_annotations)
    
    # Normalise data and define data augmentations
    train_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),])
    test_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),])
    # Load datasets
    trainset = ChalearnDataset(train_annotations, train_data_dir, transform=train_transform)
    valset = ChalearnDataset(val_annotations, valid_data_dir, transform=test_transform)
    testset = ChalearnDataset(test_annotations, test_data_dir, transform=test_transform)    
    print(len(trainset), "samples in training dataset.")
    print(len(valset), "samples in validation dataset.")
    print(len(testset), "samples in test dataset.")
    
    # Define Hyperparametersearch space
    config ={
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(8, 11)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
        # "l3": tune.sample_from(lambda _: 2 ** np.random.randint(5, 7)),
        "lr": tune.loguniform(7e-5, 7e-3),
        # "dropout": tune.loguniform(0.3, 0.7),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([128, 256, 512]),
    }

    # Define Raytune Hyperparameter optimization algorithm
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    
    # Define Raytune Reporter
    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss",
                        "train_correlation_O", "val_correlation_O",
                        "train_correlation_C", "val_correlation_C",
                        "train_correlation_E", "val_correlation_E",
                        "train_correlation_A", "val_correlation_A",
                        "train_correlation_N", "val_correlation_N",
                        "accuracy_O", "accuracy_C", "accuracy_E", "accuracy_A", "accuracy_N", "total_mean_accuracy",
                        "training_iteration"], metric='val_loss', mode='min')
    
    # Model training and Hyperparameter optimization
    print("Entering Hyperparameter tuning loop now.")
    print("Training with model architecture: {}".format(model_name))
    result = tune.run(
        partial(train_model,
                model_name=model_name,
                trainset=trainset, 
                valset=valset,
                max_num_epochs=max_num_epochs
               ),
        name = experiment_name,
        local_dir='runs/',
        keep_checkpoints_num=1, checkpoint_score_attr="min-val_loss",
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        fail_fast=True)
    
    # Now let's fetch the best model and evaluate it on our hold out test dataset
    best_trial = result.get_best_trial(metric="val_loss", mode="min", scope="all") ## scope='all' DOES NOT WORK as expected
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation Pearson correlation for Openness: {}".format(
        best_trial.last_result["val_correlation_O"]))
    print("Best trial final validation Pearson correlation for Conscientiousness: {}".format(
        best_trial.last_result["val_correlation_C"]))
    print("Best trial final validation Pearson correlation for Extraversion: {}".format(
        best_trial.last_result["val_correlation_E"]))
    print("Best trial final validation Pearson correlation for Agreableness: {}".format(
        best_trial.last_result["val_correlation_A"]))
    print("Best trial final validation Pearson correlation for Neuroticism: {}".format(
        best_trial.last_result["val_correlation_N"]))
    
    # Load model architecture
    best_trained_model = load_model(model_name, best_trial.config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    
    # Load model state dict
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    print("Checkpoint dir of best model: ", best_checkpoint_dir)
    
    # Evaluate model on test dataset
    test_loss, pearson_test_correlation_O, pearson_test_correlation_C, pearson_test_correlation_E, pearson_test_correlation_A, pearson_test_correlation_N, accuracy_O, accuracy_C, accuracy_E, accuracy_A, accuracy_N, total_mean_accuracy = get_test_statistics(
        testset=testset, model=best_trained_model, device=device, best_dir=best_checkpoint_dir)
    print("Best trial test MSE: {}".format(test_loss))
    
    print("Best trial test Pearson correlation for Openness: {}".format(pearson_test_correlation_O))
    print("Best trial test Pearson correlation for Conscientiousness: {}".format(pearson_test_correlation_C))
    print("Best trial test Pearson correlation for Extraversion: {}".format(pearson_test_correlation_E))
    print("Best trial test Pearson correlation for Agreableness: {}".format(pearson_test_correlation_A))
    print("Best trial test Pearson correlation for Neuroticism: {}".format(pearson_test_correlation_N))
    
    print("Best trial test accuracy for Openness: {}".format(accuracy_O))
    print("Best trial test accuracy for Conscientiousness: {}".format(accuracy_C))
    print("Best trial test accuracy for Extraversion: {}".format(accuracy_E))
    print("Best trial test accuracy for Agreableness: {}".format(accuracy_A))
    print("Best trial test accuracy for Neuroticism: {}".format(accuracy_N))
    print("Best trial test total mean accuracy: {}".format(total_mean_accuracy))
    
    # Add suffix to folder that contains the best_trained_model checkpoint
    default_folder = '/'.join(best_checkpoint_dir.split('/')[:-2]) + '/'
    best_model_folder = '/'.join(best_checkpoint_dir.split('/')[:-3]) + '/best_model_' + best_checkpoint_dir.split('/')[-3] + '/'
    os.rename(default_folder, best_model_folder)    


if __name__ == "__main__":
    # Read user input
    num_samples = int(sys.argv[1])
    max_num_epochs = int(sys.argv[2])
    experiment_name = str(sys.argv[3])
    model_name = str(sys.argv[4])
    # Run training and evaluation
    main(num_samples=num_samples, max_num_epochs=max_num_epochs, gpus_per_trial=1,
         experiment_name=experiment_name, model_name=model_name)
    print("Program finished.")
