""" 
    Autor: Joshua Guerrero
    Fecha: 03/03/2023
    -----------------------

    Conv1 -----> Clase para generar un red neuronal convolucional
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"


import torch
import pandas as pd
import torch.nn as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Conv1(torch.nn.Module):
    def __init__(self, input_data, len_y_output):
        self.kernel_size = 1
        self.padding = 0
        self.stride = 1
        super().__init__()
        self.entrada = input_data
        self.len_tags = len_y_output
        self.conv1 = torch.nn.Conv1d(
            self.entrada, self.get_out_layer(self.entrada), kernel_size=self.kernel_size, stride=self.stride)
        self.conv2 = torch.nn.Conv1d(
            self.conv1.out_channels, self.get_out_layer(self.conv1.out_channels), kernel_size=self.kernel_size,stride=self.stride)
       

        
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        x = torch.randn(1,self.entrada , 1)
        self._to_linear = None
        self.convs(x)
        self.fc1 = torch.nn.Linear(self._to_linear, 16)
        self.fc2 = torch.nn.Linear(16, self.len_tags)


    def get_out_layer(self,in_data):
        
        l_out = int(((in_data + (2 * self.padding) * (self.kernel_size - 1) - 1 ) / self.stride) + 1)
        
        return l_out

    def convs(self,x):
        x = F.max_pool1d(F.relu(self.conv1(x)),1)
        x = self.dropout(x)
        x = F.max_pool1d(F.relu(self.conv2(x)),1)
        x = self.dropout(x)
    


        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]
            print(x[0].shape)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.softmax(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# Función principal para entrenar la red neuronal convolucional 

def main(device,filename):
    epochs = 10_000
    batch_size = 1000
    print("----------------------------------------------------------")
    print("Dividimos el dataset generar en dos datasets (Train, Test)")

    dirname = os.getcwd()
    #data = pd.read_csv(f"{dirname}\\parts\\dictionary_ngrams5_matrix.csv")
    data = pd.read_pickle(f"{dirname}/tesis/ngrams/{filename}")

    count = 0
    data["nota_clinica"] = pd.Series()
    for index, rows in data.iterrows():
        data.at[index, "nota_clinica"] = rows.values[1:-2]
        count += 1
    print(count)
    # Nos quedamos con las columnas de "Notas clínicas" y "Etiquetado ICD (Llenando valores nullos con la etiqueta "NoDerma")"
    X = data["nota_clinica"]
    y = data["tag_icd"].fillna(value="NoDerma")
    unique_attr = list(set(y))
    tag_to_num = {tag: i + 1 for i, tag in enumerate(unique_attr)}
    indices = [tag_to_num[attr] for attr in y]
    tensor_tags = torch.tensor(indices)
    one_hot_tags = torch.nn.functional.one_hot(tensor_tags, num_classes=-1)
    # Train Size
    size_train_test = .75
    X_train_size = int(len(X) * size_train_test)
    X_test_size = len(X) - X_train_size

    # Output
    # One hot encoding
    y_train_size = int(len(one_hot_tags) * size_train_test)
    y_test_size = len(one_hot_tags) - y_train_size

    # Train and Test set
    X_train, X_test = random_split(X, [X_train_size, X_test_size])
    y_train, y_test = random_split(one_hot_tags, [y_train_size, y_test_size])
    X_train_tensor = torch.Tensor(X_train.dataset)
    x_test_tensor = torch.Tensor(X_test.dataset)

    print("----------------------------------------------------------")
    # longitud de cada vector "input_data_conv"
    input_data_conv = X_train_tensor.shape[1]
    len_y_data_conv = y_train.dataset.shape[1]
    print("Creando una red convolucional para clasificación de notas")
    net = Conv1(input_data_conv, len_y_data_conv).to(device)
    print(net)
    train(net, epochs, batch_size, X_train_tensor, y_train.dataset, device)
    return


def train(net, epochs, batch_size, X_train_tensor, y_train_tensor, device):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    loss_ep = []
    print(len(X_train_tensor))
    for epoch in range(epochs):
        for i in tqdm(range(0, len(X_train_tensor), batch_size)):
            """ if(i+batch_size > len(X_train_tensor)):
                break """
            #optimizer.zero_grad()

            batch_x = X_train_tensor[i:i+batch_size].to(device)
            batch_y = y_train_tensor[i:i+batch_size].to(device)
            net.zero_grad()
            batch_x = batch_x.view(len(batch_x), batch_x.shape[1], 1)
            output = net(batch_x)            
            batch_y = batch_y.view(-1,11).float()
            loss = F.cross_entropy(output, batch_y)
            loss.backward()

            optimizer.step()
        loss_ep.append(loss.cpu().detach().numpy())
        print(f"Epoch: {epoch}. LOSS: {loss}")

    # Crear la gráfica
    epochs = range(0, len(loss_ep))
    plt.plot(epochs, loss_ep, label='Entropy Loss')

    # Agregar etiquetas y título
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Gráfica de y = x^2')

    # Agregar una leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.grid(True)
    plt.savefig('/001/usuarios/joshuaguerrero/tesis/mi_grafica.png')

    plt.show()
# MAIN program

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")

        # Llamamos a la función principal
        # paso 1
        main(device, "tf_produccion_dictionary_ngrams2_matrix.pkl")
    else:
        print("sin GPU")
