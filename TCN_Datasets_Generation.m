clc; clearvars; close all; warning('off','all');

% Load pre-defined DNN Testing Indices
load('./samples_indices_18000.mat');

configuration = 'testing'; % training or testing
% Define Simulation parameters
nSC_In = 104; % 52 for real, 52 for imag
nSC_Out = 96; % 48 for real, 48 for imag
nSym = 50; % 50 for real, 50 for imag

mobility = 'High';
modu = '16QAM';
ChType = 'VTV_SDWW';
scheme = 'DPA_TA';

ppositions = [7, 21, 32, 46].'; % Pilot positions
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].'; % Data positions

if isequal(configuration, 'training')
    indices = training_samples;
    EbN0dB = 40;
elseif isequal(configuration, 'testing')
    indices = testing_samples;
    EbN0dB = 0:5:40;
end

Dataset_size = size(indices, 1);
SNR = EbN0dB.';
N_SNR = length(SNR);

for n_snr = 1:N_SNR
    load(['./', mobility, '_', ChType, '_', modu, '_', configuration, '_simulation_', num2str(EbN0dB(n_snr)), '.mat'], ...
        'True_Channels_Structure', [scheme '_Structure'], 'HLS_Structure');

    Dataset_X = zeros(nSC_In/2, 2 * nSym, Dataset_size);
    Dataset_Y = zeros(nSC_Out/2, 2 * nSym, Dataset_size);

    % Ensure the dimensions are correct
    True_Channels_Structure = True_Channels_Structure(dpositions,:,:);
    scheme_Channels_Structure = eval([scheme '_Structure']);

    % Loop through each dataset sample
    for i = 1:Dataset_size
        % Populate Dataset_X
        Dataset_X(:, 1, i) = real(HLS_Structure(:, i)); % Real part of HLS
        Dataset_X(:, 2, i) = imag(HLS_Structure(:, i)); % Imaginary part of HLS
        for sym = 1:nSym - 1
            Dataset_X(:, 2 * sym + 1, i) = real(scheme_Channels_Structure(:, sym, i));
            Dataset_X(:, 2 * sym + 2, i) = imag(scheme_Channels_Structure(:, sym, i));
        end
        
        % Populate Dataset_Y
        for sym = 1:nSym
            Dataset_Y(:, 2 * sym - 1, i) = real(True_Channels_Structure(:, sym, i));
            Dataset_Y(:, 2 * sym, i) = imag(True_Channels_Structure(:, sym, i));
        end
    end

    % Permute dimensions for training or testing
    Dataset_X = permute(Dataset_X, [3, 2, 1]);
    Dataset_Y = permute(Dataset_Y, [3, 2, 1]);

    if isequal(configuration, 'training')
        TCN_Datasets.('Train_X') = Dataset_X;
        TCN_Datasets.('Train_Y') = Dataset_Y;
    elseif isequal(configuration, 'testing')
        load(['./', mobility, '_', ChType, '_', modu, '_', configuration, '_simulation_', num2str(EbN0dB(n_snr)), '.mat'], ...
            'Received_Symbols_FFT_Structure');

        Received_Symbols_FFT_Structure = Received_Symbols_FFT_Structure(:,:,:);
        for i = 1:Dataset_size
            Dataset_X(:, 1, i) = real(HLS_Structure(:, i)); % Real part of HLS
            Dataset_X(:, 2, i) = imag(HLS_Structure(:, i)); % Imaginary part of HLS
            for sym = 1:nSym - 1
                Dataset_X(:, 2 * sym + 1, i) = real(scheme_Channels_Structure(:, sym, i));
                Dataset_X(:, 2 * sym + 2, i) = imag(scheme_Channels_Structure(:, sym, i));
            end

            % Populate Dataset_Y
            for sym = 1:nSym
                Dataset_Y(:, 2 * sym - 1, i) = real(True_Channels_Structure(:, sym, i));
                Dataset_Y(:, 2 * sym, i) = imag(True_Channels_Structure(:, sym, i));
            end
        end

        Dataset_X = permute(Dataset_X, [3, 2, 1]);
        Dataset_Y = permute(Dataset_Y, [3, 2, 1]);
        Received_Symbols_FFT_Structure = permute(Received_Symbols_FFT_Structure, [3, 2, 1]);

        TCN_Datasets.('Test_X') = Dataset_X;
        TCN_Datasets.('Test_Y') = Dataset_Y;
        TCN_Datasets.('Y_DataSubCarriers') = Received_Symbols_FFT_Structure;
    end

    % Save the datasets
    save(['./', mobility, '_', ChType, '_', modu, '_', scheme, '_Transformer_', configuration, '_dataset_', num2str(EbN0dB(n_snr)), '.mat'], 'TCN_Datasets');
end
