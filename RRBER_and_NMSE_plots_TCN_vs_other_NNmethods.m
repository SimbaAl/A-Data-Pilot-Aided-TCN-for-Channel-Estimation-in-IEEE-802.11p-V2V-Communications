clc; clearvars; close all; warning('off', 'all');
% Define Simulation parameters
mobility = 'High';
modu = '16QAM';
ChType = 'VTV_SDWW';

filename = ['./', mobility, '_', ChType, '_', modu, '_simulation_parameters.mat'];
loadedData = load(filename);

% Load new results
newResults = load('results_script5.mat');
% New results
ERR_scheme_DNN = newResults.ERR_scheme_DNN;
BER_scheme_DNN = newResults.BER_scheme_DNN;

% Load new results
TCNResults = load('TCN_Ber_nmse_results.mat');


% New results
Err_scheme_TCN = TCNResults.Err_scheme_TCN;
Ber_scheme_TCN = TCNResults.Ber_scheme_TCN;

% Load new results
newResults = load('STA_results.mat');
% New results
ERR_STA_DNN = newResults.ERR_STA_DNN;
BER_STA_DNN = newResults.BER_STA_DNN;

% Load new results
LSTM_MLP_Results = load('LSTM_MLP_DPA_TA.mat');
% New results
ERR_DPA_TA_LSTM_MLP = LSTM_MLP_Results.ERR_DPA_TA_LSTM_MLP;
BER_DPA_TA_LSTM_MLP = LSTM_MLP_Results.BER_DPA_TA_LSTM_MLP;

% Load new results
LSTM_MLP_noTA_Results = load('LSTM_MLP_no_TA.mat');
% New results
ERR_no_TA_LSTM_MLP = LSTM_MLP_noTA_Results.ERR_no_TA_LSTM_MLP;
BER_no_TA_LSTM_MLP = LSTM_MLP_noTA_Results.BER_no_TA_LSTM_MLP;

% Load new results
dual_cell_DPA_TA_Results = load('dual_cell_DPA_TA.mat');
% New results
ERR_dual_cell_DPA_TA = dual_cell_DPA_TA_Results.ERR_dual_cell_DPA_TA;
BER_dual_cell_DPA_TA = dual_cell_DPA_TA_Results.BER_dual_cell_DPA_TA;


BER_Ideal = loadedData.BER_Ideal;
BER_Initial = loadedData.BER_Initial;
BER_DPA_TA = loadedData.BER_DPA_TA;
BER_LS = loadedData.BER_LS;
BER_STA = loadedData.BER_STA;
BER_TRFI = loadedData.BER_TRFI;
BER_CDP = loadedData.BER_CDP;

ERR_Initial = loadedData.ERR_Initial;
ERR_LS = loadedData.ERR_LS;
ERR_DPA_TA = loadedData.ERR_DPA_TA;
ERR_TRFI = loadedData.ERR_TRFI;
ERR_CDP = loadedData.ERR_CDP;
ERR_STA = loadedData.ERR_STA;

% Plotting NMSE
figure;
semilogy(ERR_Initial, 'k--o', 'DisplayName', 'Initial');
hold on;
semilogy(ERR_LS, 'b-p', 'DisplayName', 'LS');
%semilogy(ERR_STA, 'm-v', 'DisplayName', 'STA');

%semilogy(ERR_CDP, 'r-*', 'DisplayName', 'CDP');
%semilogy(ERR_TRFI, 'g-s', 'DisplayName', 'TRFI');
semilogy(ERR_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA');
semilogy(ERR_scheme_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN'); % Add new NMSE data
semilogy(ERR_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 'x', 'DisplayName', 'STA-DNN'); % Add new NMSE data
semilogy(Err_scheme_TCN, 'color', [0.5, 0, 1], 'LineStyle', '-', 'Marker', 'h', ...
    'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.75, 0.25, 1], ...
    'DisplayName', 'TCN-(proposed)');
semilogy(ERR_DPA_TA_LSTM_MLP, 'color', [1, 0, 1], 'LineStyle', '-', 'Marker', '+', 'DisplayName', 'LSTM-MLP-DPA-TA');
semilogy(ERR_no_TA_LSTM_MLP, 'r-*', 'DisplayName', 'LSTM-MLP-no-TA');
semilogy(ERR_dual_cell_DPA_TA, 'g-s', 'DisplayName', 'Dual-Cell-DPA-TA');
hold off;
xticklabels({'0','5','10','15','20','25','30','35','40'});
xlabel('SNR (dB)');
ylabel('NMSE');
title('NMSE');
legend('Location', 'northeast');
grid on;
ylim([1e-4, 1e2]);

% Plotting BER
figure;
semilogy(BER_Ideal, 'k--o', 'DisplayName', 'Ideal');
hold on;
semilogy(BER_Initial, 'k-d', 'DisplayName', 'Initial');
semilogy(BER_LS, 'b-p', 'DisplayName', 'LS');
%semilogy(BER_STA, 'm-v', 'DisplayName', 'STA');

%semilogy(BER_CDP, 'r-*', 'DisplayName', 'CDP');
%semilogy(BER_TRFI, 'g-s', 'DisplayName', 'TRFI');
semilogy(BER_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA');
semilogy(BER_scheme_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN'); % Add new BER data
semilogy(BER_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 'x', 'DisplayName', 'STA-DNN'); % Add new NMSE data
semilogy(Ber_scheme_TCN, 'color', [0.5, 0, 1], 'LineStyle', '-', 'Marker', 'h', ...
    'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.75, 0.25, 1], ...
    'DisplayName', 'TCN-(proposed)');
semilogy(BER_DPA_TA_LSTM_MLP, 'color', [1, 0, 1], 'LineStyle', '-', 'Marker', '+', 'DisplayName', 'LSTM-MLP-DPA-TA');
semilogy(BER_no_TA_LSTM_MLP, 'r-*', 'DisplayName', 'LSTM-MLP-noTA');
semilogy(BER_dual_cell_DPA_TA, 'g-s', 'DisplayName', 'Dual-Cell-DPA-TA');
xticklabels({'0','5','10','15','20','25','30','35','40'});
hold off;
xlabel('SNR (dB)');
ylabel('BER');
title('BER');
legend('Location', 'southwest');
grid on;
ylim([1e-4, 1e0]);
