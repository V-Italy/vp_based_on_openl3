# -*- coding: utf-8 -*-
import os
import glob
import pathlib
import openl3
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics

def EmbeddingExtraction(input_folder, output_folder, input_repr="linear", cont_type="env", emb_size=512):
    """
    Creating of dataset
    
    Parameters:
    ----------
    param input_folder: Input folder name.
    param output_folder: Ouput folder name.
    param input_repr: Spectrogram representation used for model. The default is "linear".
    param cont_type: Type of content used to train the embedding model. The default is "env".
    param emb_size: Embedding dimensionality.Embedding size. The default is "512".
    ----------
    return: None
    """    
    
    directory = output_folder
    
    if emb_size != 512 or emb_size != 6144:
        emb_size = 512
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for path in glob.glob(input_folder + "/*"):
        user_id = pathlib.Path(path).name
        embeddings = np.array([]).reshape(0, emb_size)
        for file in glob.glob(path + "/**/*.flac"):
            print(f'user ID = {user_id} | file = {file}')
            audio, sr = sf.read(file)
            emb, ts = openl3.get_audio_embedding(audio, sr, input_repr=input_repr, content_type=cont_type, embedding_size=emb_size)
            embeddings = np.concatenate([embeddings, emb])
        with open(f"{directory}/{user_id}.csv", 'w') as f:
            np.savetxt(f, embeddings, delimiter=';', fmt='%f')
            print(f"user ID = {user_id} | embeddings size = {embeddings.shape}")
            print(f"user ID = {user_id} -> {directory}/{user_id}.csv")
        
       

def DatasetCreating(input_folder, output_folder, train_size=70, val_size=15, test_size=15):
    """
    Creating of dataset
    
    Parameters:
    ----------
    param input_folder: Input folder name.
    param output_folder: Ouput folder name.
    param train_size: Train data size. The default is "70".
    param val_size: Validation data size. The default is "15".
    param test_size: Test data size. The default is "15".
    ----------
    return: None
    """
    
    directory_train = output_folder + "/train/"
    if not os.path.exists(directory_train):
        os.makedirs(directory_train)
    directory_val = output_folder + "/val/"    
    if not os.path.exists(directory_val):
        os.makedirs(directory_val)      
    directory_test = output_folder + "/test/"
    if not os.path.exists(directory_test):
        os.makedirs(directory_test)   
    
    for file in glob.glob(input_folder + "/*.csv"):
        print(f'file = {file}')
        with open(file, 'r') as f_in:
            embeddings = pd.read_csv(f_in, sep=';', header=None)
            embeddings = np.array(embeddings)
            user_id = int(pathlib.Path(file).name.split('.')[0])
            
            ind = np.random.permutation(embeddings.shape[0]) 
            
            print(f"user ID = {user_id} | embeddings size = {embeddings.shape}")
            
            #split data
            embeddings_perm = embeddings[ind]
            p = np.array([train_size/100, val_size/100, test_size/100])  
            data = np.split(embeddings_perm,(embeddings_perm.shape[0]*p[:-1].cumsum()).astype(int))
           
            global embeddings_train, embeddings_test, embeddings_val
            embeddings_train = np.array(data[0])
            embeddings_val  = np.array(data[1])
            embeddings_test = np.array(data[2])
            
            print(f"user ID = {user_id} | train | size = {embeddings_train.shape}")
            print(f"user ID = {user_id} | val   | size = {embeddings_val.shape}")
            print(f"user ID = {user_id} | test  | size = {embeddings_test.shape}")
              
            with open(f"{directory_train}/{user_id}.csv", 'w') as f_out:
                np.savetxt(f_out, embeddings_train, delimiter=';', fmt='%f')
                print(f"user ID = {user_id} | train -> {directory_train}/{user_id}.csv")
            with open(f"{directory_val}/{user_id}.csv", 'w') as f_out:
                np.savetxt(f_out, embeddings_val, delimiter=';', fmt='%f')
                print(f"user ID = {user_id} | val   -> {directory_val}/{user_id}.csv")
            with open(f"{directory_test}/{user_id}.csv", 'w') as f_out:
                np.savetxt(f_out, embeddings_test, delimiter=';', fmt='%f')
                print(f"user ID = {user_id} | test  -> {directory_test}/{user_id}.csv")

"""
Loading dataset for training and validation
"""
class LoadDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.path = data_path
        self.data = []
        self.n_id = 0
        for file in glob.glob(data_path):
            print(f"File: {file}")
            with open(file, 'r') as f:
                embeddings = pd.read_csv(f, sep=';', header=None)
                embeddings = np.array(embeddings)
                y = self.n_id
                for ind in range(0, embeddings.shape[0]):
                    x = embeddings[ind,:]
                    self.data.append((x, y))
                self.n_id += 1
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        x, y = self.data[ind]
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

"""
Loading dataset for testing
"""
class LoadTestDataset(Dataset):
    def __init__(self, data_path, data_size, transform=None):
        self.path = data_path
        self.data = []
        self.data_size = data_size
        self.n_users = 0
        files = glob.glob(self.path)
        self.n_users  = len(files)
        print(self.n_users)

        y = 1 # Label True
        for file in files:
            print(f"File: {file}")
            with open(file, 'r') as f:
                embeddings = pd.read_csv(f, sep=';', header=None)
                x = np.array(embeddings)
                for ind in range(self.data_size):
                    self.data.append((x[ind + 0], x[ind + 1], y))

        y = 0 # Label False
        for ind in range(len(files)-1):
            file1 = files[ind + 0]
            file2 = files[ind + 1]
            print(f"{file1} | {file2}")
            with open(file1, 'r') as f1:
                embeddings1 = pd.read_csv(f1, sep=';', header=None)
                x1 = np.array(embeddings1)
            with open(file2, 'r') as f2:
                embeddings2 = pd.read_csv(f2, sep=';', header=None)
                x2 = np.array(embeddings2)
            for ind in range(self.data_size//2):
                self.data.append((x1[ind], x2[ind], y))

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x1, x2, y = self.data[ind]
        x1 = torch.tensor(x1).float()
        x2 = torch.tensor(x2).float()
        y = torch.tensor(y).long()
        return x1, x2, y
    
class SpeechClassificationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Parameters:
        ----------    
        input_dim: Input layer Size (number of neurons).
        output_dim: Output layer Size (number of neurons).
        hidden_dim: Hidden layer Size (number of neurons).
        ----------
        return: None
        """
        super(SpeechClassificationNetwork, self).__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.bn2  = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()
        self.fc3  = nn.Linear(hidden_dim, output_dim)
        self.out  = nn.Softmax(dim=1)
        
    def forward(self, x, get_embedding = False):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if get_embedding == True:
            return x
        x = self.fc3(x)
        #x = self.out(x)
        return x

def Train(model, optimizer, loss, epochs, TrainDataloader, ValidDataloader,train_loss_list, train_acc_list,valid_loss_list, valid_acc_list, device):
    """
    Neural network train
    
    Parameters:
    ----------    
    param model: neural network object.
    param optimizer: optimizer.
    param loss: loss function.
    param epochs: epochs number.
    param TrainDataloader: DataLoader object for train data.
    param ValidDataloader: DataLoader object for validation data.
    param train_loss_list: list of loss values on the train dataset
    param train_acc_list: list of accuracy values on the validation dataset
    param valid_loss_list: list of loss values on the validation dataset
    param valid_acc_list: list of accuracy values on the validation dataset
    param device: type of device: cpu or gpu
    ----------
    return: None
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = datetime.now()
        train_loss = []
        pred_labels = []
        true_labels = []
        model.train()
        optimizer.zero_grad()
        for train_x, train_y in TrainDataloader:
            x = train_x.to(device)
            true_y = train_y.to(device)
            pred_y = model(x)
            loss_train_res = loss(pred_y, true_y)
            batch_train_loss = loss_train_res.cpu().item()
            train_loss.append(batch_train_loss)
            loss_train_res.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred_labels += pred_y.argmax(axis=-1).flatten().tolist()
            true_labels += true_y.tolist()
        train_loss_mean = np.mean(train_loss)
        train_acc = accuracy_score(true_labels, pred_labels)
        train_loss_list.append(train_loss_mean)
        train_acc_list.append(train_acc)

        valid_loss = []
        pred_labels = []
        true_labels = []
        model.eval()
        with torch.no_grad():                           
            for valid_x, valid_y in ValidDataloader:
                x = valid_x.to(device)
                true_y = valid_y.to(device)
                pred_y = model(x)
                loss_valid_res = loss(pred_y, true_y)
                batch_val_loss = loss_valid_res.cpu().item()
                valid_loss.append(batch_val_loss)
                pred_labels += pred_y.argmax(axis=-1).flatten().tolist()
                true_labels += true_y.tolist()
        valid_loss_mean = np.mean(valid_loss)
        valid_acc = accuracy_score(true_labels, pred_labels)
        valid_loss_list.append(valid_loss_mean)
        valid_acc_list.append(valid_acc)
        print(f"Train Loss: {train_loss_mean:.5f}, Train Accuracy = {train_acc}")
        print(f"Valid Loss: {valid_loss_mean:.5f}, Valid Accuracy = {valid_acc}")
        print(f"Time: {datetime.now() - start_time}")



def PlotData(train_data, valid_data, type_data, file_name):
    """
    Visualization and saving of results
    
    Parameters:
    ----------    
    param train_data: results on the training dataset. 
    param valid_data: results on the validating dataset. 
    param type_data: loss or accuracy.
    param file_name: name of file.
    ----------
    return: None
    """
    plt.figure(figsize=(10,8))
    plt.plot(range(1, len(train_data)+1, 1), train_data, label='Train ' + type_data)
    plt.plot(range(1, len(valid_data)+1, 1), valid_data, label='Valid ' + type_data)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel(type_data, fontsize=16)
    plt.legend(fontsize=16)
    # Turn on the minor TICKS, which are required for the minor GRID
    plt.minorticks_on()
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig(type_data + '_' + file_name + '.png') 
    plt.show()

def EvalEmbeddings(model, TestDataloader, device):
    """
    Embedding comparison
    
    Parameters:
    ----------
    param model: trained model
    param TestDataloader: DataLoader object for test data
    param device: type of device: cpu or gpu
    ----------
    return: 
        true_labels: true labels. 
        pred_labels: predicted labels.  
    """
    model.eval()
    pred_labels = []
    true_labels = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad(): 
        for x1, x2, true_y in TestDataloader:
           true_labels += true_y.tolist()
           x1 = x1.to(device)
           x2 = x2.to(device)
           emb1 = model(x1, get_embedding = True)
           emb2 = model(x2, get_embedding = True)
           pred_y = cos(emb1.cpu(), emb2.cpu())  
           pred_labels += pred_y.tolist()
    return true_labels, pred_labels 


def compute_eer(label, pred, positive_label=1):
    """
    Python compute equal error rate (eer)
    ONLY tested on binary classification
    https://github.com/YuanGongND/python-compute-eer/blob/master/compute_eer.py
    
    Parameters:
    ----------        
    param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
    param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
    param positive_label: the class that is viewed as positive class when computing EER
    ----------    
    return: equal error rate (EER)
    """
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def main():
    data_type = "linear"#"mel128" #"linear", "mel256"
    
    needDataPrepare = False
    needModelTrain = True
    input_folder = "train-clean-100"
    output_folder = "embeddings"+"_" + data_type
    batch_size = 1024
    if needDataPrepare == True:
        EmbeddingExtraction(input_folder, output_folder, input_repr=data_type)
        input_folder = output_folder
        output_folder = "dataset_"+data_type
        DatasetCreating(input_folder, output_folder)
        

    PATH = "SCNModel" + "_" + data_type+ ".pt"
    output_folder = "dataset" + "_" + data_type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if needModelTrain == True:
        start_time = datetime.now()
        
        #Load Datasets
        TrainDataset = LoadDataset(output_folder + "/train/*.csv")
        ValidDataset = LoadDataset(output_folder + "/val/*.csv")
        TrainDataloader = DataLoader(dataset = TrainDataset, batch_size=batch_size, shuffle=True)
        ValidDataloader = DataLoader(dataset = ValidDataset, batch_size=batch_size, shuffle=True)
        print(datetime.now() - start_time)
        
        input_dim = 512
        output_dim = TrainDataset.n_id
        hidden_dim = 512
        
        SCNModel = SpeechClassificationNetwork(input_dim, output_dim, hidden_dim).to(device)
          
        learning_rate = 0.001
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(SCNModel.parameters(), lr=learning_rate)
          
        num_epochs = 20
       
        train_loss_list = []
        train_acc_list  = []
        valid_loss_list = []
        valid_acc_list  = []
          
        Train(SCNModel, optimizer, loss, num_epochs, TrainDataloader, ValidDataloader, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, device)
      
        torch.save(SCNModel, PATH)
        print('Model is saved')    

        PlotData(train_loss_list, valid_loss_list, type_data='Loss', file_name=data_type)
        PlotData(train_acc_list, valid_acc_list, type_data='Accuracy', file_name=data_type)

    else:
        SCNModel = torch.load(PATH)
        
    TestDataset = LoadTestDataset(output_folder + "/test/*.csv", data_size=64 )
    TestDataloader = DataLoader(dataset = TestDataset, batch_size=64, shuffle=True)
    true_labels, pred_labels = EvalEmbeddings(SCNModel, TestDataloader, device)
    eer = compute_eer(true_labels, pred_labels) 
    print(f"EER = {eer}")  
   
if __name__=='__main__':
    main()