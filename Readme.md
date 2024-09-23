This repository contain the code of the TCN-based estimators proposed in "A Data Pilot-Aided Temporal Convolutional Network for Channel Estimation in IEEE 802.11p Vehicle-to-Vehicle Communications [1]" published at the SATNAC 2024  that was held from 6 - 9 October at Skukuza, Kruger National Park (South Africa). The dataset is generated in Matlab and the proiposed deep learning estimators are implemented in python (PyTorch).

Files Description
Main.m: The main simulation file, where the simulation parameters are specified.
Channel_functions.m: Includes the pre-defined vehicular channel models [3] for different mobility conditions.
DPA_TA.m: Includes the implementation of the data-pilot aided (DPA) channel estimation followed by temporal averaging (TA).
TCN_Datasets_Generation.m: Generating the TCN training/testing datasets.
TCN_Results_Processing.m: Processing the testing results genertead by the TCN testing and caculate the BER and NMSE results of the TCN_based estimator.
TCN_DPA_TA.py: The TCN training/testing is performed employing the generated training/testing datasets. The file should be executed twice as follows:

Step1: Training by executing this command python TCN_DPA_TA.py Mobility Channel_Model Modulation_Order Scheme Training_SNR Mode Epochs Batch_size
ex: python Fixed_1.py High VTV_SDWW 16QAM DPA 40 train 200 128

Step2: Testing by executing this command: python TCN_DPA_TA.py Mobility Channel_Model Modulation_Scheme Scheme Testing_SNR Mode
ex: python W_A.py High VTV_SDWW 16QAM DPA 40 test

Running Steps:
To generate the dataset for the simulation, follow these steps:
1. Run the IDX_Generation.m script to generate the dataset indices, training dataset size, and testing dataset size. 

2. Run the main.m file twice:
First, set the configuration parameter to 'training' to generate the training simulation file.
Second, set the configuration parameter to 'testing' to generate the testing simulation files.
Specify the simulation parameters like the number of OFDM symbols, channel model, mobility scenario, modulation order, SNR range, and the path to the generated indices from step 1.
The generated simulation files will be saved in your working directory.

3. Run the TCN_Datasets_Generation.m file twice, similar to step 2:
First, set the configuration to 'training' and specify the channel estimation scheme and OFDM simulation parameters to generate the TCN training dataset.
Second, set the configuration to 'testing' and repeat the same to generate the TCN testing dataset.

4. Run the TCN_DPA_TA.py file twice:
First, to perform the training of the TCN model.
Second, to perform the testing of the TCN model.

5. After the training and testing, the TCN results will be saved as .mat files. Finally, run the TCN_Results_Processing.m file to get the NMSE and BER results for the studied channel estimation scheme.


References
[1] Ngorima S.A, Helberg A.S.J, Davel M.H: A Data Pilot-Aided Temporal Convolutional Network for Channel Estimation in IEEE 802.11p Vehicle-to-Vehicle Communications. In: Southern Africa Telecommunication Networks and Applications Conference (SATNAC) (2024).

For more information and questions, please contact me on aldringorima@gmail.com
