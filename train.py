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
        


def train_model(config, model_name, trainset, valset, max_num_epochs, experiment_name='Test'):
    
    log_dir='runs/' + str(experiment_name)
    writer = SummaryWriter(log_dir=log_dir, max_queue=50, flush_secs=120)
    model_path = log_dir + '/'
    
    model = load_model(model_name, config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
            params = list(model.module.fc.parameters())
    else:
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
    
    last_acc_O = 0
    last_acc_C = 0
    last_acc_E = 0
    last_acc_A = 0
    last_acc_N = 0
    
    torch.save(model,model_path + 'O' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_acc_O)}")
    torch.save(model,model_path + 'C' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_acc_C)}")
    torch.save(model,model_path + 'E' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_acc_E)}")
    torch.save(model,model_path + 'A' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_acc_A)}")
    torch.save(model,model_path + 'N' + '_' + f"epoch{0}_step{0}_corr{'%.4f'%(last_acc_N)}")
    
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
        start = time.time()
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

                    
                    # Save train and validation predictions exemplarily for Openness
                    train_path = os.path.join(model_path, "train_results_epoch_" + str(epoch) + ".csv")
                    train_df = pd.DataFrame(data={'Labels':train_labels[:,0].tolist(),
                                                  'Predictions':train_scores[:,0].tolist()})
                    train_df.to_csv(train_path, index=False)
                    val_path = os.path.join(model_path, "val_results_epoch_" + str(epoch) + ".csv")
                    val_df = pd.DataFrame(data={'Labels':val_labels[:,0].tolist(), 
                                                'Predictions':val_scores[:,0].tolist()})
                    val_df.to_csv(val_path, index=False)
                    
                    
                    ## Save models if new maximum validation accuracy was reached for any trait
                    if accuracy_O > last_acc_O:
                        last_acc_O, last_steps_O, last_epoch_O = save_model_per_trait(last_acc_O, last_steps_O, last_epoch_O,
                                                                                         accuracy_O, steps, epoch+1,
                                                                                         model_path, model, 'O')
                    if accuracy_C > last_acc_C:
                        last_acc_C, last_steps_C, last_epoch_C = save_model_per_trait(last_acc_C, last_steps_C, last_epoch_C,
                                                                                         accuracy_C, steps, epoch+1,
                                                                                         model_path, model, 'C')
                    if accuracy_E > last_acc_E:
                        last_acc_E, last_steps_E, last_epoch_E = save_model_per_trait(last_acc_E, last_steps_E, last_epoch_E,
                                                                                         accuracy_E, steps, epoch+1,
                                                                                         model_path, model, 'E')
                    if accuracy_A > last_acc_A:
                        last_acc_A, last_steps_A, last_epoch_A = save_model_per_trait(last_acc_A, last_steps_A, last_epoch_A,
                                                                                         accuracy_A, steps, epoch+1,
                                                                                         model_path, model, 'A')
                    if accuracy_N > last_acc_N:
                        last_acc_N, last_steps_N, last_epoch_N = save_model_per_trait(last_acc_N, last_steps_N, last_epoch_N,
                                                                                         accuracy_N, steps, epoch+1,
                                                                                         model_path, model, 'N')

                # Average loss across all minibatches
                train_loss = train_loss/print_every
                val_loss = val_loss/len(valloader)
              
                # Write Loss to Tensorboard
                writer.add_scalar('Loss/train', train_loss, global_step=epoch+1)
                writer.add_scalar('Loss/val', val_loss, global_step=epoch+1)
                
                # Write Pearson Correlations to Tensorboard
                writer.add_scalar('Pearson_r_Opennness/train', pearson_train_correlation_O, global_step=epoch+1)
                writer.add_scalar('Pearson_r_Opennness/val', pearson_val_correlation_O, global_step=epoch+1)

                writer.add_scalar('Pearson_r_Conscientiousness/train', pearson_train_correlation_C, global_step=epoch+1)
                writer.add_scalar('Pearson_r_Conscientiousness/val', pearson_val_correlation_C, global_step=epoch+1)

                writer.add_scalar('Pearson_r_Extraversion/train', pearson_train_correlation_E, global_step=epoch+1)
                writer.add_scalar('Pearson_r_Extraversion/val', pearson_val_correlation_E, global_step=epoch+1)

                writer.add_scalar('Pearson_r_Agreeableness/train', pearson_train_correlation_A, global_step=epoch+1)
                writer.add_scalar('Pearson_r_Agreeableness/val', pearson_val_correlation_A, global_step=epoch+1)

                writer.add_scalar('Pearson_r_Neuroticism/train', pearson_train_correlation_N, global_step=epoch+1)
                writer.add_scalar('Pearson_r_Neuroticism/val', pearson_val_correlation_N, global_step=epoch+1)
                
                # Write "Accuracy" to Tensorboard
                writer.add_scalar('val_acc_O/train', accuracy_O, global_step=epoch+1)
                writer.add_scalar('val_acc_C/train', accuracy_C, global_step=epoch+1)
                writer.add_scalar('val_acc_E/train', accuracy_E, global_step=epoch+1)
                writer.add_scalar('val_acc_A/train', accuracy_A, global_step=epoch+1)
                writer.add_scalar('val_acc_N/train', accuracy_N, global_step=epoch+1)
                writer.add_scalar('val_acc_total/train', total_mean_accuracy, global_step=epoch+1)
                
                run_time = time.time() - start
                print(f"Epoch {epoch+1}/{max_num_epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {val_loss:.3f}.. "
                  f"Elapsed time [s]: {run_time:.1f}.. "
                     )
                
                train_loss = 0
                model.train()
        
    print("Finished training.")
    writer.flush()


def main(max_num_epochs=10, experiment_name='Test', model_name='ResNet18_v1.0', batch_size=1, final_run=False):
    
    # Set data related paths
    if final_run:
        script_path = os.path.dirname(os.path.abspath( __file__ ))
        train_data_dir = script_path + '/' + 'data/train/train_and_valid_faces/'
        test_data_dir = script_path + '/' + 'data/test/faces/'
        train_labels_pickle = script_path + '/' + 'data/train/annotation_training.pkl'
        valid_labels_pickle = script_path + '/' + 'data/valid/annotation_validation.pkl'
        test_labels_pickle = script_path + '/' + 'data/test/annotation_test.pkl'        
    else:
        script_path = os.path.dirname(os.path.abspath( __file__ ))
        train_data_dir = script_path + '/' + 'data/train/faces/'
        valid_data_dir = script_path + '/' + 'data/valid/faces/'
        test_data_dir = script_path + '/' + 'data/test/faces/'
        train_labels_pickle = script_path + '/' + 'data/train/annotation_training.pkl'
        valid_labels_pickle = script_path + '/' + 'data/valid/annotation_validation.pkl'
        test_labels_pickle = script_path + '/' + 'data/test/annotation_test.pkl'
    
    # Set hyperparameters
    lr = 0.0001
    weight_decay = 0.0

    # Load label pickle files
    train_annotations = pickle.load(open(train_labels_pickle, "rb" ), encoding="latin1" )
    val_annotations = pickle.load(open(valid_labels_pickle, "rb" ), encoding="latin1")
    test_annotations = pickle.load(open(test_labels_pickle, "rb" ), encoding="latin1")
    
    # Normalise data and define data augmentations
    train_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),])
    test_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),])
    # Load datasets
    if final_run:
        train_annotations = format_label_list(train_annotations)
        val_annotations = format_label_list(val_annotations)
        train_annotations = {**train_annotations, **val_annotations}
        test_annotations = format_label_list(test_annotations)
        trainset = ChalearnDataset(train_annotations, train_data_dir, transform=train_transform)
        valset = ChalearnDataset(test_annotations, test_data_dir, transform=test_transform)
        
    else:
        train_annotations = format_label_list(train_annotations)
        val_annotations = format_label_list(val_annotations)
        trainset = ChalearnDataset(train_annotations, train_data_dir, transform=train_transform)
        valset = ChalearnDataset(val_annotations, valid_data_dir, transform=test_transform)
    
    print(len(trainset), "samples in training dataset.")
    print(len(valset), "samples in validation dataset.")
    
    # Define Hyperparametersearch space
    config ={
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(8, 11)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
        # "l3": tune.sample_from(lambda _: 2 ** np.random.randint(5, 7)),
        "lr": lr,
        # "dropout": tune.loguniform(0.3, 0.7),
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }

    # Model training
    print("Training with model architecture: {}".format(model_name))
    train_model(config, 
                model_name=model_name,
                trainset=trainset, 
                valset=valset, 
                max_num_epochs=max_num_epochs, 
                experiment_name=experiment_name)


if __name__ == "__main__":
    # Read user input
    max_num_epochs = int(sys.argv[1])
    experiment_name = str(sys.argv[2])
    model_name = str(sys.argv[3])
    final_run = sys.argv[4].lower() == 'true'
    batch_size = int(sys.argv[5])
    # Run training and evaluation
    main(max_num_epochs=max_num_epochs, experiment_name=experiment_name, model_name=model_name,
         batch_size=batch_size, final_run=final_run)
    print("Program finished.")
