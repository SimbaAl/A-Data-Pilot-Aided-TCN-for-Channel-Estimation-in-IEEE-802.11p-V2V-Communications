clc; clearvars; close all; warning('off', 'all');
% Define Simulation parameters
mobility = 'High';
modu = '16QAM';
ChType = 'VTV_SDWW';

filename = ['./', mobility, '_', ChType, '_', modu, '_simulation_parameters.mat'];
loadedData = load(filename);

% Load new results
%newResults = load('results_script5.mat');
% New results
%ERR_scheme_DNN = newResults.ERR_scheme_DNN;
%BER_scheme_DNN = newResults.BER_scheme_DNN;

% Load new results
TCNResults = load('TCN_Ber_nmse_results.mat');

% New results
Err_scheme_TCN = TCNResults.Err_scheme_TCN;
Ber_scheme_TCN = TCNResults.Ber_scheme_TCN;

% Load new results
%newResults = load('STA_results.mat');
% New results
%ERR_STA_DNN = newResults.ERR_STA_DNN;
%BER_STA_DNN = newResults.BER_STA_DNN;


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
semilogy(ERR_Initial, 'k--o', 'DisplayName', 'DPA');
hold on;
semilogy(ERR_LS, 'b-p', 'DisplayName', 'LS');
semilogy(ERR_STA, 'm-v', 'DisplayName', 'STA');
semilogy(ERR_CDP, 'r-*', 'DisplayName', 'CDP');
semilogy(ERR_TRFI, 'g-s', 'DisplayName', 'TRFI');
semilogy(ERR_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA');
%semilogy(ERR_scheme_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN'); % Add new NMSE data
%semilogy(ERR_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'STA-DNN'); % Add new NMSE data
semilogy(Err_scheme_TCN, 'color', [0.6, 0, 0.4], 'LineStyle', '-', 'Marker', 'h', ...
    'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0.75, 0.25, 1], ...
    'DisplayName', 'TCN-DPA-TA-(baseline)');
xticklabels({'0','5','10','15','20','25','30','35','40'});
hold off;
xlabel('SNR (dB)');
ylabel('NMSE');
title('NMSE');
legend('Location', 'northeast');
grid on;
%ylim([1e-4, 1e2]);

% Plotting BER
figure;
semilogy(BER_Ideal, 'k--o', 'DisplayName', 'Ideal');
hold on;
semilogy(BER_Initial, 'k-d', 'DisplayName', 'DPA');
semilogy(BER_LS, 'b-p', 'DisplayName', 'LS');
semilogy(BER_STA, 'm-v', 'DisplayName', 'STA');
semilogy(BER_CDP, 'r-*', 'DisplayName', 'CDP');
semilogy(BER_TRFI, 'g-s', 'DisplayName', 'TRFI');
semilogy(BER_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA');
%semilogy(BER_scheme_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN'); % Add new BER data
%semilogy(BER_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 'x', 'DisplayName', 'STA-DNN'); % Add new NMSE data
%semilogy(Ber_scheme_TCN, 'color', [0.5, 0, 1], 'LineStyle', '-', 'Marker', 'h', 'DisplayName', 'TCN-(proposed)'); % New NMSE data in purple
semilogy(Ber_scheme_TCN, 'color', [0.6, 0, 0.4], 'LineStyle', '-', 'Marker', 'h', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.75, 0.25, 1], ...
    'DisplayName', 'TCN-DPA-TA-(baseline)');
xticklabels({'0','5','10','15','20','25','30','35','40'});
hold off;
xlabel('SNR (dB)');
ylabel('BER');
title('BER');
legend('Location', 'southwest');
grid on;
ylim([1e-4, 1e0]);
