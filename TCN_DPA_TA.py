## 104 subcarriers are used in this script instead of 96 that are used in the No_Scaler,py script. Everything else is the same
#in this script: I have repeated everything done in the close_.py script except i have chosen to remove a standard scaler

#I have used data subcarriers as time steps which is the second dimension. ( 16000, 96,50)
# I have corrected the post processing: 
'''For the first OFDM symbol (j == 0), the channel estimate hf_DL_TCN[i, j, :] is initialized with the output of the TCN model hf_out.
For the remaining OFDM symbols (j > 0), we follow the DPA procedure:
y_eq = yf_d[i, j, :] / hf_DL_TCN[i, j-1, :] computes the equalized received symbol using the previous channel estimate.
q = fn.map(fn.demap(y_eq, modu_way), modu_way) maps the equalized received symbol to the closest constellation point.
hf_DL_TCN[i, j, :] = yf_d[i, j, :] / q updates the channel estimate using the received signal and the constellation point.
This implementation follows the equations (3) and (4) from the theory TCN paper.
in addition i have added TA, which I will be removing if DPA performs enough.
'''

'''Transposed X shape: (16000, 96, 50)
Transposed Y shape: (16000, 96, 50)
Rearranged X shape: (16000, 96, 50)
Press Enter to continue...
Reshaped Training Input Dataset:  (38400000, 2)
Reshaped Training Label Dataset:  (38400000, 2)

Reshaped Normalized Training Input Dataset:  (16000, 96, 50)
Reshaped Normalized Training Label Dataset:  (16000, 96, 50)
dataset size:  16000 , train set size:  12000 , val set size:  4000
Train_X : (12000, 96, 50)
Train_Y : (12000, 96, 50)
Val_X : (4000, 96, 50)
Val_Y : (4000, 96, 50)'''



import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy
import scipy.io
import pickle
import functions as fn
import sys
import os
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from fastai.basics import *

# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)
train_rate = 0.75
val_rate = 0.25

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.000001):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        #self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        #self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        #self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        #self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.conv1.weight)
        init.zeros_(self.conv1.bias)
        init.xavier_uniform_(self.conv2.weight)
        init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            init.xavier_uniform_(self.downsample.weight)
            init.zeros_(self.downsample.bias)

    '''def init_weights(self):
        init.xavier_uniform_(self.conv1.weight, gain=init.calculate_gain('tanh'))
        init.xavier_uniform_(self.conv2.weight, gain=init.calculate_gain('tanh'))
        if self.downsample is not None:
            init.xavier_uniform_(self.downsample.weight, gain=init.calculate_gain('tanh'))'''

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        #return self.relu(out + res)
        return out + res

    '''def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        comb = (out + res)
        return self.relu(out + res)
        #return comb'''


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=8, dropout=0.000001):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            if i == num_levels - 1:  # Add the additional Conv1d layer after the last TemporalBlock
                layers += [nn.Conv1d(out_channels, out_channels, 1, groups=out_channels)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
            

def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error


def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse
    
num_inputs = 50  
num_channels = [50, 50]  
model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=8, dropout=0.000001)
#model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels)

# Training phase
if configuration_mode == 13:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    dnn_input = sys.argv[6]
    hidden_layer1 = sys.argv[7]
    hidden_layer2 = sys.argv[8]
    hidden_layer3 = sys.argv[9]
    dnn_output = sys.argv[10]
    epoch = sys.argv[11]
    #BATCH_SIZE = sys.argv[9]
    BATCH_SIZE = int(sys.argv[12])

    mat = loadmat('./{}_{}_{}_{}_TCN_training_dataset.mat'.format(mobility, channel_model, modulation_order, scheme))
    Dataset = mat['TCN_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)  
    print('Loaded Dataset Outputs: ', Y.shape)
    #input("Press Enter to continue...")  
    
    # Transpose the datasets to bring subcarriers to the second dimension
    #X = np.transpose(X, (0, 2, 1))
    #Y = np.transpose(Y, (0, 2, 1))

    #print("Transposed X shape:", X.shape)
    #print("Transposed Y shape:", Y.shape)
    #input("Press Enter to continue...")
    
    # Splitting the dataset into real and imaginary parts
    #real_part = X[:, :48, :]  # First 52 are real
    #imag_part = X[:, 48:, :]  # Last 52 are imaginary

    # Interleaving real and imaginary parts
    #X_new = np.empty((X.shape[0], 96, X.shape[2]))
    #X_new[:, ::2, :] = real_part
    #X_new[:, 1::2, :] = imag_part
    #print("Rearranged X shape:", X_new.shape)
    #input("Press Enter to continue...")


    # Reshape Input and Label Data
    input_data_Re = X.reshape(-1, 2)
    label_data_Re = Y.reshape(-1, 2)
    print('Reshaped Training Input Dataset: ', input_data_Re.shape)
    print('Reshaped Training Label Dataset: ', label_data_Re.shape)

    # Normalization
    scaler = StandardScaler()
    input_data_sclar = scaler.fit_transform(input_data_Re)  # .reshape(input_data.shape)
    label_data_sclar = scaler.fit_transform(label_data_Re)  # .reshape(label_data.shape)

    # Reshape after normalization
    input_data_sclar = input_data_sclar.reshape(X.shape)
    label_data_sclar = label_data_sclar.reshape(Y.shape)
    print('Reshaped Normalized Training Input Dataset: ', input_data_sclar.shape)
    print('Reshaped Normalized Training Label Dataset: ', label_data_sclar.shape)

    # Training and Validation Datasets splits
    nums = X.shape[0]
    train_nums = int(train_rate * nums)
    val_nums = int(nums * val_rate)
    print('dataset size: ', nums, ', train set size: ', train_nums, ', val set size: ', val_nums)

    # Assign training data set and validation data set
    Train_X = input_data_sclar[:train_nums]
    Train_Y = label_data_sclar[:train_nums]
    Val_X = input_data_sclar[-val_nums:]
    Val_Y = label_data_sclar[-val_nums:]
    print('Train_X :', Train_X.shape)
    print('Train_Y :', Train_Y.shape)
    print('Val_X :', Val_X.shape)
    print('Val_Y :', Val_Y.shape)
    train_input = torch.from_numpy(Train_X).type(torch.FloatTensor)
    train_label = torch.from_numpy(Train_Y).type(torch.FloatTensor)
    val_input = torch.from_numpy(input_data_sclar[-val_nums:]).type(torch.FloatTensor)
    val_label = torch.from_numpy(label_data_sclar[-val_nums:]).type(torch.FloatTensor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ---------------- generate batch dataset ------------------- #
    dataset = data.TensorDataset(train_input, train_label)
    loader = data.DataLoader(dataset=dataset, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=8 if torch.cuda.is_available() else 0)


    # Loss and optimizer
    #criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # setting learning rate to 0.0001
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    epoch = int(sys.argv[11])
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(loader), epochs=100)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(loader), epochs=epoch)
    #optimizer = optim.Adam(model.parameters())
    
    # Reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.01, patience=10, min_lr=0.000001)

    # Variables for early stopping
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_loss = float('inf')
    
    
    model_path = './{}_{}_{}_{}_TCN_{}.pt'.format(mobility, channel_model, modulation_order, scheme, training_snr)

    training_losses = []
    validation_losses = []
    total_training_time = 0
    best_val_error = float('inf')
    best_val_loss = float('inf')

    for ep in range(int(epoch)):
    
        start_time = time.time()
        model.train()
        total_loss = 0
        total_train_error = 0

        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_err, _, _ = calc_error(outputs.cpu().detach().numpy(), batch_y.cpu().detach().numpy())
            total_train_error += avg_err

        avg_train_loss = total_loss / len(loader)
        avg_train_error = total_train_error / len(loader)
        

         
        #if step % 200 == 0:
            #print(f'Epoch [{ep+1}/{epoch}], Training Loss: {avg_train_loss:.4f}, Avg Error: {avg_train_error:.4f}')
        training_losses.append(avg_train_loss)
        
        
       
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_errors = []
        with torch.no_grad():
                                    
            val_input, val_label = val_input.to(device), val_label.to(device)
            val_outputs = model(val_input)
            val_loss = criterion(val_outputs, val_label).item()
            
            
            avg_err, _, _ = calc_error(val_outputs.cpu().numpy(), val_label.cpu().numpy())
            val_errors = [avg_err]
            
            # Inverse scaling
            val_outputs_inv = scaler.inverse_transform(val_outputs.cpu().numpy().reshape(-1, 2)).reshape(val_outputs.shape)
            val_y_inv = scaler.inverse_transform(val_label.cpu().numpy().reshape(-1, 2)).reshape(val_label.shape)
            
            # Calculate error on inverse scaled data
            #avg_error, step_error, error = calc_error(val_outputs, val_label)
            
            validation_losses.append(val_loss)


        
        #print(f'Epoch [{ep+1}/{epoch}], Training Loss: {avg_train_loss:.4f}, Training Error: {avg_train_error:.8f}, Validation Loss: {val_loss:.4f}, Validation Error: {avg_error:.8f}')
        print(f'Epoch [{ep+1}/{epoch}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        #scheduler.step()
         
 
            #print(f'Epoch [{ep+1}/{epoch}], Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path) 
        
        else:
             early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {ep+1}")
            break
        
                
            # Check for best model
            #if avg_error < best_val_error:
               #best_val_error = avg_error
               #torch.save(model.state_dict(), model_path)
               
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        #total_training_time += time.time() - start_time
                
    print("Training completed. Total time: {:.2f} seconds.".format(total_training_time))
            
    
    #print("Training completed.")
    
    # Plotting
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  # Save the plot as a .png file


# Testing phase
else:
    # We are running the testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]
    if modulation_order == 'QPSK':
        modu_way = 1
    elif modulation_order == '16QAM':
        modu_way = 2
    
    num_ofdm_symbols = 50
    num_subcarriers = 96
    num_channels = [50, 50]
    
    #num_inputs = 50
    #num_channels = [50, 50] 
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize and load the TCN model
    model_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_{testing_snr}.pt'
    #model = TemporalConvNet(num_inputs=96, num_channels=num_channels, kernel_size=48, dropout=0.0002)
    model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=8, dropout=0.000001)
    #model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=2, dropout=0.2)
    #model = TemporalConvNet(num_inputs, num_channels=num_channels, kernel_size=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    scaler = StandardScaler()
 
    model.eval()

    
    
    with torch.no_grad():         
        for n_snr in SNR_index:
            mat = loadmat(f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_testing_dataset_{n_snr}.mat')
            Dataset = mat['TCN_Datasets']
            Dataset = Dataset[0, 0]
            X_original = Dataset['Test_X']
            Y = Dataset['Test_Y']
            yf_d = Dataset['Y_DataSubCarriers']
            print('Loaded Dataset Inputs: ', X_original.shape)
            print('Loaded Dataset Outputs: ', Y.shape)
            print('Loaded Testing OFDM Frames: ', yf_d.shape)

            # Transpose the datasets to bring subcarriers to the second dimension
            #X = np.transpose(X_original, (0, 2, 1))
            #Y = np.transpose(Y_original, (0, 2, 1))

            #print("Transposed X shape:", X.shape)
            #print("Transposed Y shape:", Y.shape)

            # Splitting the dataset into real and imaginary parts
            #real_part = X[:, :48, :]  # First 52 are real
            #imag_part = X[:, 48:, :]  # Last 52 are imaginary

            #print("Real part shape:", real_part.shape)
            #print("Imaginary part shape:", imag_part.shape)
            #input("Press Enter to continue...")


            # Interleaving real and imaginary parts
            #X_new = np.empty((X.shape[0], 96, X.shape[2]))
            #X_new[:, ::2, :] = real_part
            #X_new[:, 1::2, :] = imag_part
            #print("Rearranged X shape:", X_new.shape)
            
            #hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
            #print("Shape of hf_DL_TCN: ", hf_DL_TCN.shape)

            X = X_original
            
            hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
            print("Shape of hf_DL_TCN: ", hf_DL_TCN.shape)

            for i in range(yf_d.shape[0]):
                print(f'Processing Frame | {i}')
                initial_channel_est = X[i, 0, :]
                initial_channel_est = scaler.fit_transform(initial_channel_est.reshape(-1, 2)).reshape(initial_channel_est.shape)
                input_data = np.tile(initial_channel_est, (50, 1))
                input_tensor = torch.from_numpy(input_data).type(torch.FloatTensor).unsqueeze(0)
                #input_tensor = torch.from_numpy(X[i]).type(torch.FloatTensor).unsqueeze(0)
                output_tensor = model(input_tensor.to(device))  # (1, 50, 96)
                
                output_data = scaler.inverse_transform(output_tensor.detach().cpu().numpy().reshape(-1, 2)).reshape(output_tensor.shape)

                for j in range(yf_d.shape[1]):
                    hf_out = output_data[0, j, :48] + 1j * output_data[0, j, 48:]
                    #hf_out = output_data[0, :48, j] + 1j * output_data[0, 48:, j]  # (50,)
                    #real_part = output_data[0, :, j][:48]
                    #imag_part = output_data[0, :, j][48:]
                    #hf_out = real_part + 1j * imag_part  # Convert to complex values                    
                   
                    if j == 0:
                        hf_DL_TCN[i, j, :] = hf_out
                    else:
                        y_eq = yf_d[i, j, :] / hf_DL_TCN[i, j - 1, :]
                        q = fn.map(fn.demap(y_eq, modu_way), modu_way)
                        hf_DL_TCN[i, j, :] = yf_d[i, j, :] / q
                    
                    if j < yf_d.shape[1] - 1:
                        # Update the initial channel estimate for the next OFDM symbol
                        updated_channel_est = np.concatenate((hf_DL_TCN[i, j, :].real, hf_DL_TCN[i, j, :].imag)).ravel()
                        #updated_channel_est = np.concatenate((hf_out.real, hf_out.imag)).ravel()
                        X[i, :,j + 1] = updated_channel_est
                        initial_channel_est = 0.5 * initial_channel_est + 0.5 * X_new[i, :,j + 1]

            
            '''hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[2], yf_d.shape[1]), dtype="complex64")
            #hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[2], yf_d.shape[1]), dtype="complex64")
            print("Shape of hf_DL_TCN: ", hf_DL_TCN.shape)

            #device = torch.device("cpu")
            for i in range(yf_d.shape[0]):
                #hf = X[i, 0, :]
                print(f'Processing Frame | {i}')
                initial_channel_est = X[i, :, 0]
                initial_channel_est = scaler.fit_transform(initial_channel_est.reshape(-1, 2)).reshape(initial_channel_est.shape)
                #input_data = np.broadcast_to(initial_channel_est[:, np.newaxis], (initial_channel_est.shape[0], yf_d.shape[1]))
                #input_tensor = torch.from_numpy(input_data.copy()).type(torch.FloatTensor).unsqueeze(0)

                input_tensor = torch.from_numpy(initial_channel_est).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                #input_data = np.tile(initial_channel_est, (yf_d.shape[1], 1)).T
                #input_tensor = torch.from_numpy(input_data).type(torch.FloatTensor).unsqueeze(0)
                output_tensor = model(input_tensor.to(device)) # (1, 50, 96)
                output_data = scaler.inverse_transform(output_tensor.detach().cpu().numpy().reshape(-1, 2)).reshape(output_tensor.shape)
                
                output_data = output_tensor.detach().cpu().numpy().reshape(output_tensor.shape)
                
                for j in range(yf_d.shape[2]):
                    hf_out = output_data[0, :, j]  # (50,)
                    #hf_out = output_data[0, :48, j] + 1j * output_data[0, 48:, j]  # (50,)
                    if j == 0:
                    
                        hf_DL_TCN[i, :, j] = hf_out
                    else: 
                        y_eq = yf_d[i, :, j] / hf_DL_TCN[i, :, j-1]
                        #sf = yf_d[i, j, :] / hf_out  # (48,) # Equalize the received signal using the previous estimate
                        q = fn.map(fn.demap(y_eq, modu_way), modu_way)  # Map the equalized signal to the constellation
                        hf_DL_TCN[i, : j] = yf_d[i, : j] / q  # Update the channel estimate using the equalized signal and the constellation point'''

                    #if j < yf_d.shape[1] - 1:
                        # Update the initial channel estimate for the next OFDM symbol
                        #updated_channel_est = np.concatenate((hf_DL_TCN[i, j, :].real, hf_DL_TCN[i, j, :].imag)).ravel()
                        #updated_channel_est = np.concatenate((hf_out.real, hf_out.imag)).ravel()
                        #X_new[i, :,j + 1] = updated_channel_est
                        #initial_channel_est = 0.5 * initial_channel_est + 0.5 * X_new[i, :,j + 1]

            # Save Results
            result_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_Results_{n_snr}.pickle'
            dest_name = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_Results_{n_snr}.mat'
            with open(result_path, 'wb') as f:
                pickle.dump([X, Y, hf_DL_TCN], f)
            scipy.io.savemat(dest_name, {f'{scheme}_TCN_test_x_{n_snr}': X,
                                         f'{scheme}_TCN_test_y_{n_snr}': Y,
                                         f'{scheme}_TCN_predicted_y_{n_snr}': hf_DL_TCN})
            print("Data successfully converted to .mat file")
            os.remove(result_path)
            


                           
#hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
#This line initializes an empty NumPy array hf_DL_TCN to store the predicted channel estimates for each frame, OFDM symbol, and subcarrier
#iterates over each frame in the testing dataset.
##prepares the input data for the current frame by:
#Retrieving the initial channel estimate from X for the current frame.
#Scaling the initial channel estimate using the scaler object.
#Tiling the initial channel estimate 50 times (for 50 OFDM symbols).
#Converting the input data to a PyTorch tensor and moving it to the appropriate device.
#Passing the input tensor through the TCN model to obtain the output tensor.
#Inversely scaling the output tensor using the scaler object.

##updates the initial channel estimation hf_out for the next frame using the output from the last OFDM symbol of the current frame. It involves:

#Extracting the real and imaginary parts of the output for the last OFDM symbol.
#Dividing the true data subcarriers by hf_out to obtain the soft symbols (sf).
#Demapping the soft symbols to obtain the estimated symbols (x).
#Mapping the estimated symbols back to the modulation constellation (xf).
#Updating hf_out by dividing the true data subcarriers by xf.

##This loop stores the predicted channel estimates for each OFDM symbol of the current frame in hf_DL_TCN.

#This section updates the initial channel estimate in X for the next frame using hf_out obtained from the last OFDM symbol of the current frame. It involves:

#Concatenating the real and imaginary parts of hf_out.
#Flattening the concatenated array to match the shape of X[i+1, 0, :].
#Assigning the flattened array to X[i+1, 0, :].


#In this post processing, Instead of processing the input frame symbol-by-symbol, the entire input frame (with shape (50, 96)) is preprocessed and passed through the TCN model in one forward pass.
# The output of the TCN model is expected to have the shape (1, 50, 96), so the post-processing is adapted to handle this output format.
#The post-processing loop now iterates over the OFDM symbols (dimension 1) of the output tensor, instead of the input frame symbols.

