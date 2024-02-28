""" 
    Autor: Joshua Guerrero
    Fecha: 03/03/2023
    -----------------------

    Conv1 -----> Clase para generar un red neuronal convolucional
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import pandas as pd

import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Conv1(torch.nn.Module):
    def __init__(self,input_data,len_y_output):
        super().__init__()
        self.entrada = input_data
        self.len_tags = len_y_output
        self.conv1 = torch.nn.Conv1d(self.entrada,100,1,1,1)
        self.conv2 = torch.nn.Conv1d(self.conv1.out_channels,10,1,1,1)
        self.droput = torch.nn.Dropout()
        x = torch.randn(self.entrada).view(1,self.entrada,1)
        self._to_linear = None
        self.convs(x)
        self.fc1 = torch.nn.Linear(self._to_linear,16)
        self.fc2 = torch.nn.Linear(16,self.len_tags)
      
    def convs(self,x):
        input_data = x
        conv_layer_1 = self.conv1(input_data)

        # size of the matrix changes from 998 to 3
        relu_layer = F.relu(conv_layer_1)
        plot = torch.tensor(relu_layer).detach().numpy()
        #plt.plot(plot[0])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        col = plot[:,0]
        ax.scatter(plot[:,0],plot[:,1],plot[:,2])
        plt.show()


        x = F.max_pool1d(relu_layer,1)
        x = F.max_pool1d(F.relu(self.droput(self.conv2(x))),1)
        x = F.max_pool1d(F.relu(self.droput(self.conv2(x))),3)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

def main(device):
    
    print("----------------------------------------------------------")
    print("Dividimos el dataset generar en dos datasets (Train, Test)")

    dirname = os.getcwd()
    #data = pd.read_csv(f"{dirname}\\parts\\dictionary_ngrams5_matrix.csv")
    data = pd.read_csv(f"{dirname}/tesis/parts/dictionary_ngrams5_matrix.csv")
    
    count = 0
    data["nota_clinica"] = pd.Series()
    for index,rows in data.iterrows():
        data["nota_clinica"][count] = rows.values[1:-2]
        count += 1


    # Nos quedamos con las columnas de "Notas clínicas" y "Etiquetado ICD (Llenando valores nullos con la etiqueta "NoDerma")"
    X = data["nota_clinica"]
    y = data["tag_icd"].fillna(value="NoDerma")
    unique_attr = list(set(y))
    tag_to_num = {tag: i + 1 for i, tag in enumerate(unique_attr)}
    indices = [tag_to_num[attr] for attr in y]
    tensor_tags = torch.tensor(indices)
    one_hot_tags = torch.nn.functional.one_hot(tensor_tags,num_classes =-1)
    # Train Size
    size_train_test = .75
    X_train_size = int(len(X) * size_train_test)
    X_test_size = len(X) - X_train_size
    
    #Output
    # One hot encoding
    y_train_size = int(len(one_hot_tags) * size_train_test)
    y_test_size = len(one_hot_tags) - y_train_size
    
    
    # Train and Test set
    X_train, X_test = random_split(X, [X_train_size, X_test_size])
    y_train, y_test = random_split(one_hot_tags, [y_train_size, y_test_size])
    X_train_tensor = torch.Tensor(X_train.dataset)
    x_test_tensor = torch.Tensor(X_test.dataset)
    Y_train_tensor = torch.Tensor(y_train.dataset)
    y_train_tensor = torch.Tensor(y_test.dataset)
    print("----------------------------------------------------------")

    print("Creando una red convolucional para clasificación de notas")
    net = Conv1(X_train_tensor.shape[1], len(tag_to_num)).cuda()
    print(net)
    train(net,1,1,X_train_tensor,Y_train_tensor)
    return

def train(net, epochs, batch_size,X_train_tensor,y_train_tensor):
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    #loss_function = F.binary_cross_entropy()
    loss_ep = []
    for epoch in range(epochs):
        for i in tqdm(range(0,len(X_train_tensor),batch_size)):
            
            #batch_x = X_train_tensor[i:i+batch_size]
            #batch_y = y_train_tensor[i:i+batch_size]

            batch_x, batch_y = X_train_tensor.cuda(), y_train_tensor.cuda()
            
            net.zero_grad()
            #batch_x = batch_x.view(len(batch_x),batch_x.shape[1],1)
            output = net(batch_x)
            print(output.shape)
            batch_y = batch_y.view(-1,1)
            
            
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            
            optimizer.step()
        
        loss_ep.append(loss)
        print(f"Epoch: {epoch}. LOSS: {loss}")


# MAIN program

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")

        # Llamamos a la función principal
        # paso 1
        main(device)