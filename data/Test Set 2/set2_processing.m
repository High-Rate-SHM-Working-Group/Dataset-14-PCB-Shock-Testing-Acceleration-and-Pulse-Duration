clc;
clear variables;
close all;

%% Extract Data
DataFolder = 'C:\Users\AFRL_USER\Documents\2025 - Ryan Yount\Test Set 2\m018x1+T_p6KG\'; % choose unit to read

MainFolder = cd;
cd(DataFolder) % step into folder

PFfiles0 = dir('t0*.txt');
PFfiles1 = dir('t1*.txt');
PFfiles2 = dir('t2*.txt');

PFfiles = [PFfiles0;PFfiles1;PFfiles2];

numfiles = size(PFfiles,1);

testnum = 1;
[~,index] = sortrows({PFfiles.datenum}.'); PFfiles = PFfiles(index); clear index %sort files by date to match order of tests recorded in testnotes
% AccelFSIN = input('Enter Acceleration FSIN = ');
AccelFSIN=30000;
FSOUT = 10;

R = struct(); % output struct

for k = 1:length(PFfiles)
    PFfilename = PFfiles(k).name;
    PFdata = dlmread(PFfilename,'\t',8,0); %read file skipping header see example in MATLAB folder
    time = PFdata(:,end);
    Accel = PFdata(:,1).*AccelFSIN./FSOUT;
    Board = PFdata(:,2).*AccelFSIN./FSOUT;
    resistor = PFdata(:,3);
    N = length(time);
    Fs = 1/mean(diff(time));
    freq = Fs*(0:N/2)/N;

    if any(diff(time)<=0)
        [time, uniqueIdx] = unique(time, 'sorted');
        Accel = Accel(uniqueIdx);
    end
    WidthPercent = 10;
    [peak, peakIdx] = max(Accel);
    w = pulsewidth(Accel,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peak]);

    % filtering
    fc = 50e3; % Low-pass filter pass freq
    Fs = 1/mean(diff(time));
    [b,a] = butter(2, fc/(Fs/2), 'low');
    AccelFiltered = filtfilt(b,a,Accel);

    [peak, peakIdx] = max(AccelFiltered);
    WidthPercent = 10;
    [wF,initcross, finalcross,midlevel] = pulsewidth(AccelFiltered,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peak]);

    % velocity
    IdxEnd = find((Accel(peakIdx:end)<0),1)+peakIdx;
    vel = trapz(time(1:IdxEnd),Accel(1:IdxEnd)*32.2); %[ft/s]

    Accel_0 = zeros(1e6, 1);
    Accel_0(1:length(Accel)) = Accel;
    Board_0 = zeros(1e6, 1);
    Board_0(1:length(Board)) = Board;
    T=1/Fs;
    N=length(Accel_0);
    freq = Fs*(0:N/2)/N;

    Accelfft = computeFFT(Accel_0, N);
    Boardfft = computeFFT(Board_0, N);

    % Save results
    R(k).Filename = PFfilename;
    R(k).Time = time;
    R(k).Accel = Accel;
    R(k).Board = Board;
    R(k).Res = resistor;
    R(k).AccelFiltered = AccelFiltered;
    R(k).Peak = peak;
    R(k).Pulsewidth = wF;
    R(k).Velocity = vel;
    R(k).Freq = freq;
    R(k).AccelFFT = Accelfft;
    R(k).BoardFFT = Boardfft;
end

%% change back to main folder
cd(MainFolder)

%% Plot time domain

% figure;
% t = tiledlayout(2,1);
% title(t, 'Time Domain');
% ax1 = nexttile;
% hold on;
% yyaxis left
% for i = 1:length(PFfiles)
%     plot(R(i).Time, R(i).Accel, LineWidth=1.5)
% end
% ylabel('Base')
% yyaxis right
% plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2)
% ylabel('Voltage')
% ylim([-3 3]);
% ax2 = nexttile;
% hold on;
% yyaxis left
% for i = 1:length(PFfiles)
%     plot(R(i).Time, R(i).Board, LineWidth=1.5)
% end
% ylim([-1e4 1e4])
% ylabel('Board')
% yyaxis right
% plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2)
% ylabel('Voltage')
% ylim([-3 3]);
% linkaxes([ax1 ax2],'x')
% xlim([])
base_ylim = [-7500 7500];
board_ylim = [-7500 7500];
xlimit = [0.0095 0.013];
figure;
t = tiledlayout(2,1);
title(t, 'Time Domain');
ax1 = nexttile;
hold on;
yyaxis left
plot(R(length(PFfiles)).Time, R(length(PFfiles)).Accel, LineWidth=1.5)
ylabel('Base (g_n)')
ylim(base_ylim)
yyaxis right
plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2)
ylabel('Voltage')
ylim([-3 3]);
box;
ax2 = nexttile;
hold on;
yyaxis left
plot(R(length(PFfiles)).Time, R(length(PFfiles)).Board, LineWidth=1.5)
ylim(board_ylim)
ylabel('Board (g_n)')
yyaxis right
plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2)
ylabel('Voltage')
ylim([-3 3]);
box;
linkaxes([ax1 ax2],'x')
xlim(xlimit);

% figure;
% t = tiledlayout(2,1);
% title(t, 'Time Domain Plots');
% 
% ax1 = nexttile;
% hold on;
% yyaxis left
% h = [];
% for i = 1:length(PFfiles)
%     h(i) = plot(R(i).Time, R(i).Accel, LineWidth=1.5);
% end
% 
% last_color = h(end).Color;
% 
% yyaxis right
% if isequal(last_color, [1 0 0])
%     right_color = 'b';
% else
%     right_color = 'r';
% end
% plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2, Color=right_color)
% 
% ax2 = nexttile;
% hold on;
% yyaxis left
% for i = 1:length(PFfiles)
%     plot(R(i).Time, R(i).Board, LineWidth=1.5)
% end
% ylim([-2e4 2e4])
% ylabel('Board')
% yyaxis right
% plot(R(length(PFfiles)).Time, R(length(PFfiles)).Res, LineWidth=2)
% ylabel('Voltage')
% ylim([-3 3]);
% linkaxes([ax1 ax2],'x')


%% Plot FFTs
limx = [0 4500];
figure;
f = tiledlayout(2,1);
title(f, 'FFTs');
ax1 = nexttile;
hold on;
for i = 1:length(PFfiles)
    plot(R(i).Freq, abs(R(i).AccelFFT), LineWidth=1.5)
end
ylabel('Base')
ax2 = nexttile;
hold on;
for i = 1:length(PFfiles)
    plot(R(i).Freq, abs(R(i).BoardFFT), LineWidth=1.5)
end
ylabel('Board')
linkaxes([ax1 ax2],'x')
xlim(limx);

plots = [];
leg = {};
figure;
hold on;
for i = 1:length(PFfiles)
    plots(i) = plot(R(i).Freq, abs(R(i).BoardFFT), LineWidth=1);
    leg{i} = sprintf('Impact %d', i);
end
ylabel('Acceleration |g_n|');
legend(plots, leg);
xlabel('Frequency (Hz)');
xlim(limx);
grid on;
box;

%% Calculate FRFs

file_start = 1;
file_end = length(PFfiles);
file_num = file_end - (file_start-1);
accelffts = zeros(length(Accelfft), file_num); 
boardffts = zeros(length(Boardfft), file_num);


for i = file_start:file_end
    accelffts(:,i) = R(i).AccelFFT;
    boardffts(:,i) = R(i).BoardFFT;
end                                              

[Gxy, Gyx, Gyy, Gxx] = computeSpectralDensities(accelffts, boardffts);
[coher, Hw] = computeHwCoherence(Gxy, Gyx, Gyy, Gxx);

%% Plot FRF

% figure;
% f = tiledlayout(2,1);
% title(f, 'FRF and Coherence')
% ax1 = nexttile;
% plot(R(1).Freq, abs(Hw), 'DisplayName', 'Base/Board')
% hold on;
% ylabel('FRF');
% legend('show');
% ax2 = nexttile;
% plot(R(1).Freq, coher)
% hold on;
% ylabel('Coherence');
% linkaxes([ax1 ax2],'x')
% xlim(limx);

%% Functions

function rfft = computeFFT(input, N)
input_w_full = fft(input)/N;

input_w = input_w_full(1:N/2+1,:);
input_w(2:end-1,:) = 2*input_w(2:end-1,:);

rfft = input_w;
end

function [Gxy, Gyx, Gyy, Gxx] = computeSpectralDensities(inputFFT, responseFFT)

Gxy = sum((inputFFT).*conj(responseFFT),2);
Gyx = sum((responseFFT).*conj(inputFFT),2);
Gyy = sum(conj(responseFFT).*responseFFT,2);
Gxx = sum(conj(inputFFT).*inputFFT,2);

end

function [coher, Hw] = computeHwCoherence(Gxy, Gyx, Gyy, Gxx)

H1= Gyx./Gxx; % Assumes noise only on output/will capture antiresonance better  
H2 = Gyy./Gxy; % Assumes noise only on input/will capture resonance better

Hw = (H1+H2)./2; % FRF to minimize noise on input and output

coher = H1./H2;
% coher= ((Gxy.*Gyx)./(Gxx.*Gyy));

end