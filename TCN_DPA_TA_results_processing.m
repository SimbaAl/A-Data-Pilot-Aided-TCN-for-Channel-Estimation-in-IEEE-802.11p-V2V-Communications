clc;clearvars;close all; warning('off','all');

mobility = 'High';
ChType = 'VTV_SDWW';
modu = '16QAM';
scheme = 'DPA';
testing_samples = 2000;
if(isequal(modu,'QPSK'))
nBitPerSym       = 2; 
elseif (isequal(modu,'16QAM'))
nBitPerSym       = 4; 
elseif (isequal(modu,'64QAM'))
 nBitPerSym       = 6; 
end
M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM    
load(['./',mobility,'_',ChType,'_',modu,'_simulation_parameters']);
EbN0dB                    = (0:5:40)'; 
nSym                      = 50;
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
scramInit                 = 93;
nDSC                      = 48;
nUSC                      = 52;
dpositions                = [1:6, 8:20, 22:31, 33:45, 47:52].'; 
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
N_SNR                     = size(EbN0dB,1);
Phf                       = zeros(N_SNR,1);
Err_scheme_TCN           = zeros(N_SNR,1);
Ber_scheme_TCN           = zeros(N_SNR,1);

for n_snr = 1:N_SNR 
    tic;
    % Loading Simulation Parameters Results
    load(['./',mobility,'_',ChType,'_',modu,'_testing_simulation_',num2str(EbN0dB(n_snr)),'.mat']);
    % Loading scheme-DNN Results
    %load(['./',mobility,'_',ChType,'_',modu,'_',scheme,'_LSTM_Results_',num2str(EbN0dB(n_snr)),'.mat']); 
    load(['./',mobility,'_',ChType,'_',modu,'_',scheme,'_TCN_Results_',num2str(EbN0dB(n_snr)),'.mat']);

     %TestY = eval([scheme,'_TCN_test_y_',num2str(EbN0dB(n_snr))]);
     %TestY = eval([scheme,'_test_y_',num2str(EbN0dB(n_snr))]);
     %TestY = permute(TestY,[3 2 1]);
     %TestY = TestY(1:nDSC,:,:) + 1i*TestY(nDSC+1:2*nDSC,:,:);
     
     scheme_TCN = eval([scheme,'_TCN_predicted_y_',num2str(EbN0dB(n_snr))]);
     %scheme_LSTM = eval([scheme,'_corrected_y_',num2str(EbN0dB(n_snr))]);
     scheme_TCN = permute(scheme_TCN,[3 2 1]);
     True_Channels_Structure = True_Channels_Structure(:,:,:); 
     for u = 1:size(scheme_TCN,3)
 
        H_scheme_TCN = scheme_TCN(dpositions ,:,u);

        Phf(n_snr) = Phf(n_snr) + mean(sum(abs(True_Channels_Structure(:,:,u)).^2)); 
        Err_scheme_TCN (n_snr) =  Err_scheme_TCN (n_snr) +  mean(sum(abs(H_scheme_TCN - True_Channels_Structure(dpositions,:,u)).^2)); 
        
        % IEEE 802.11p Rx
        Bits_scheme_TCN     = de2bi((qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions ,:,u) ./ H_scheme_TCN),M)));
        %Bits_AE_DNN     = de2bi((qamdemod(sqrt(Pow) * (EqualizedS(:,:,u) ),M)));
        Ber_scheme_TCN(n_snr)   = Ber_scheme_TCN(n_snr) + biterr(wlanScramble(vitdec((matintrlv((deintrlv(Bits_scheme_TCN(:),Random_permutation_Vector)).',Interleaver_Columns,16).'),poly2trellis(7,[171 133]),34,'trunc','hard'),93),TX_Bits_Stream_Structure(:,u));
     end
   toc;
end
Phf = Phf ./ testing_samples;
Err_scheme_TCN = Err_scheme_TCN ./ (testing_samples * Phf); 
Ber_scheme_TCN = Ber_scheme_TCN/ (testing_samples * nSym * 48 * nBitPerSym);

%% Save the data in a .mat file
% Save the results
%filename = 'TCN_BER_NMSE_results.mat';
%save(filename, 'BER_scheme_TCN', 'ERR_scheme_TCN', 'Phf');
save('TCN_Ber_nmse_results.mat', 'Phf', 'Ber_scheme_TCN', 'Err_scheme_TCN');
%% Assuming you have already calculated Phf, ERR_scheme_DNN, and BER_scheme_DNN

% Plot NMSE_scheme_DNN
%NMSE_scheme_DNN = ERR_scheme_LSTM ./ Phf; % Calculate NMSE from ERR_scheme_DNN and Phf
figure;
plot(EbN0dB, Err_scheme_TCN, 'r-s');
xlabel('SNR (dB)');
ylabel('Normalized Mean Squared Error (NMSE)');
title('Normalized MSE TCN vs SNR');
grid on;

% Plot BER_scheme_DNN
figure;
plot(EbN0dB, Ber_scheme_TCN, 'g-d');
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('Bit Error Rate (BER\_scheme\TCN) vs SNR');
grid on;