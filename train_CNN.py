import copy
import os
import pdb
import sys
import argparse

import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import SubsetRandomSampler

import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def transform_images(train_folder):
    data_transform = transforms.Compose([
        transforms.Resize(256),        # 1. Redimensiona para garantir que a menor aresta tenha 256 pixels
        transforms.CenterCrop(224),    # 2. Corta a parte central para 224x224
        transforms.ToTensor()          # 3. Converte para Tensor
    ])

    images_transformed = datasets.ImageFolder(os.path.join(train_folder), transform=data_transform)

    return images_transformed

# Setting up the model for training purposes
def setting_model(cnn_model_name, pre_trained):
    if cnn_model_name == 'resnet50':
        if pre_trained == 1:
          cnn_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
          cnn_model = models.resnet50(weights=None)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device) # Device from main (GPU or CPU)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2) # 2 because it's binary classification

    elif cnn_model_name == 'vgg16':
        if pre_trained == 1:
          cnn_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
          cnn_model = models.vgg16(weights=None)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, 2)

    elif cnn_model_name == 'googlenet':
        if pre_trained == 1:
            cnn_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        else:
            cnn_model = models.googlenet(weights=None)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2)

    else:
        print('Invalid model')
        sys.exit(1)

    #cnn_model = cnn_model.cuda()

    return cnn_model

def training_parameters(learning_rate):
    criteria = torch.nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimize, step_size=15, gamma=0.75)

    return criteria, optimize, lr_scheduler

def store_scores(pos_probabilities, validation_labels):
    # Dataframe to save the scores obtained from validation
    val_scores = pd.DataFrame(columns=['score', 'class'])

    # Appending the scores with their respective classes in the dataframe
    for val_proba_list, val_label_list in zip(pos_probabilities, validation_labels):
        for list_probabilities, list_labels in zip(val_proba_list, val_label_list):
            for pos_prob, label in zip(list_probabilities, list_labels):

                # Using only 4 decimal digits
                pos_prob = float("{:.4f}".format(pos_prob))

                instance = {'score': pos_prob, 'class': label}
                instance = pd.DataFrame([instance])
                
                val_scores = pd.concat([val_scores, instance], ignore_index=True)

    return val_scores

def validate_model(model_cnn, criterion, val_dataloader, best_eval_loss):

    eval_acc = 0.0

    # Lists of the probabilities and it's respective class from validation
    pos_probabilities = []
    validation_labels = []

    model_cnn.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_cnn(inputs)

            _, predictions = torch.max(outputs, 1)

            # Clipping the probabilities [0, 1]
            probabilities = torch.nn.functional.softmax(outputs, 1)

            pos_probabilities.append(probabilities[:,1].tolist()) # To obtain only the number from the tensor
            validation_labels.append(labels.tolist()) # To obtain only the number from the tensor

            eval_acc += torch.sum(predictions == labels.data)
            eval_loss = criterion(outputs, labels)

    eval_acc /= len(val_dataloader.dataset)
    eval_loss /= len(val_dataloader.dataset)

    print(f'Validation set accuracy = {round(float(eval_acc), 4)}')
    print(f'Validation set loss = {round(float(eval_loss), 4)}')

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        print(f'Validation best loss updated {round(float(eval_loss), 4)}')

    return pos_probabilities, validation_labels, best_eval_loss


def train_model(model_cnn, criterion_train, optimizer_train, scheduler_train, epochs, train_dataloaders):

    best_loss = float('inf')

    total_train_size = 0
    for dataloader in train_dataloaders:
        total_train_size += len(dataloader.dataset)

    model_cnn.train()
    for epoch in range(epochs):
        acc = 0.0
        loss = 0.0

        for train_dataloader in train_dataloaders:

            # Obtaining images and respective labels from the folder at batch's size
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero gradients for training
                optimizer_train.zero_grad()

                outputs = model_cnn(inputs)

                if model_name == 'googlenet' and pt == 0:
                    outputs = outputs.logits

                _, predictions = torch.max(outputs, 1)
                loss = criterion_train(outputs, labels)

                # Update model parameters in the direction that minimizes the loss
                loss.backward()
                optimizer_train.step()
                scheduler_train.step()

                # Counting the hits and misses from the predictions
                acc += torch.sum(predictions == labels.data)
                loss += loss.item() * inputs.size(0)

        # Calculating the epoch's accuracy and loss
        acc /= total_train_size
        loss /= total_train_size

        print(f'Epoch {epoch} accuracy = {round(float(acc), 4)}')
        print(f'Epoch {epoch} loss = {round(float(loss), 4)}')

        # Updating best model
        if loss < best_loss:
            print(f'Best model updated Epoch {epoch}: {round(float(loss), 4)}')
            best_loss = loss

    return model_cnn

if __name__ == '__main__':
    # Device for running the train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment name')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='CNN model')
    args = parser.parse_args()

    experiment_name = args.experiment
    model_name = args.model

    gs_df = None

    gs_path = os.path.abspath('gridSearch/' + experiment_name + '/' + model_name)
    if not os.path.exists(gs_path):
        os.makedirs(gs_path)

    gs_file_path = os.path.abspath(gs_path + '/gridSearch.csv')
    if os.path.exists(gs_file_path):
        gs_df = pd.read_csv(gs_file_path)
    else:
        gs_df = pd.DataFrame(columns=['lr', 'epochs', 'pt', 'batch_size', 'best_loss', 'train_scores'])

    # Getting the images and transforming it for the CNN
    image_datasets = transform_images('trains/' + experiment_name)

    fold_size = (len(image_datasets))/2
    folds = None

    if len(image_datasets) % 2 != 0:
        half_dts = int(fold_size)
        folds = [half_dts, half_dts+1]
    else:
        half_dts = int(fold_size)
        folds = [half_dts, half_dts]

    seed = torch.Generator().manual_seed(17)

    train_set = torch.utils.data.random_split(image_datasets, folds, seed)

    param_grid = {
        '1epochs': [100, 1000],
        '2pt': [0, 1],
        '3lr': [0.001, 0.01, 0.1],
        '4batch_size': [16, 32, 64]
    }

    param_combinations = list(ParameterGrid(param_grid))

    for param in tqdm(param_combinations, desc='Grid search'):
        epochs = param['1epochs']
        pt = param['2pt']
        lr = param['3lr']
        batch_size = int(param['4batch_size'])

        print(f'Training with lr: {lr}, epochs: {epochs}, pt: {pt}, batch_size: {batch_size}')

        # Creating dataloaders here so it can vary the batch_size
        dataloaders = []
        for set in train_set:
            dataloader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=0)
            dataloaders.append(dataloader)

        pos_probabilities = []
        validation_labels = []

        # Variables to pick the best weights of the model
        best_eval_loss = float('inf')

        for i in tqdm(range(2), desc='Training phase'):
            print(f"Validation with fold: {i}")

            model = setting_model(model_name, pt)
            criterion, optimizer, scheduler = training_parameters(lr)

            validation_dataloader = dataloaders.pop(i)

            model = train_model(model, criterion, optimizer, scheduler, epochs, dataloaders) 
            new_prob, new_labels, best_eval_loss = validate_model(model, criterion, validation_dataloader, best_eval_loss)

            pos_probabilities.append(new_prob)
            validation_labels.append(new_labels)

            dataloaders.insert(i, validation_dataloader)

        #validation_scores = store_scores(pos_probabilities, validation_labels)

        # Saving the validation scores into a csv for future usage
        #scores_path = os.path.abspath('scores/' + experiment_name + '/' + model_name)
        #if not os.path.exists(scores_path):
         #   os.makedirs(scores_path)

        #validation_scores_path = os.path.abspath(scores_path + '/ValidationScores_' + str(lr) + '_' + str(epochs) + '_' + str(pt) + '_' + str(batch_size) + '.csv')
        #validation_scores.to_csv(validation_scores_path, index=False)

        instance = {'lr': float(lr), 'epochs': int(epochs), 'pt': int(pt), 'batch_size': int(batch_size), 'best_loss': float(best_eval_loss)}
        instance = pd.DataFrame([instance])

        gs_df = pd.concat([gs_df, instance], ignore_index=True)
        gs_df.to_csv(gs_path + '/gridSearch.csv', index=False)

    min_loss_config_index = gs_df['best_loss'].idxmin()
    min_loss_config = gs_df.loc[min_loss_config_index]

    best_lr = min_loss_config['lr']
    best_epochs = min_loss_config['epochs']
    best_pt = min_loss_config['pt']
    pt = best_pt # Verification in line 164
    best_batch_size = int(min_loss_config['batch_size'])

    print(f'best lr: {best_lr}, best epochs: {best_epochs}, best pt: {best_pt}, best batch_size: {best_batch_size}')

    model = setting_model(model_name, best_pt)
    criterion, optimizer, scheduler = training_parameters(best_lr)

    full_train_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=best_batch_size, shuffle=True, num_workers=0)
    model = train_model(model, criterion, optimizer, scheduler, best_epochs, [full_train_dataloader])

    # Saving model weights
    model_path = os.path.abspath('models/' + experiment_name + '/')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(model.state_dict(), model_path + '/' + model_name + '_' + experiment_name + '.pth')
    gs_df.to_csv(gs_path + '/gridSearch.csv', index=False)

    print(model_name + ' for experiment ' + experiment_name + ' trained successfully')