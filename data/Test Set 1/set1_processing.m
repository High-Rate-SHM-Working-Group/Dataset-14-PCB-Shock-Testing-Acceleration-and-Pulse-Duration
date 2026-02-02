clc;
clear variables;
% close all;

%% Extract Data
DataFolder = 'C:\Users\AFRL_USER\Documents\2025 - Ryan Yount\Test Set 1\m014_p5KG\'; % choose unit to read

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
    resistor = PFdata(:,2);
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

    T=1/Fs;
    N=length(time);
    freq = Fs*(0:N/2)/N;

    Accelfft = computeFFT(Accel, N);

    % Save results
    R(k).Filename = PFfilename;
    R(k).Time = time;
    R(k).Accel = Accel;
    R(k).Res = resistor;
    R(k).AccelFiltered = AccelFiltered;
    R(k).Peak = peak;
    R(k).Pulsewidth = wF;
    R(k).Velocity = vel;
    R(k).Freq = freq;
    R(k).AccelFFT = Accelfft;
end

%% change back to main folder
cd(MainFolder)

%% Plot time domain

figure;
hold on;
for i = 1:length(PFfiles)
    plot(R(i).Time, R(i).Accel, LineWidth=1.5)
end
ylabel('Base')

figure;
hold on;
plot(R(i).Time, R(i).Accel, LineWidth=1.5)
ylabel('Base Accel (g_n)')
xlabel('Time (s)')
xlim([0.0095 0.013]);


%% Plot FFTs

figure;
hold on;
for i = 1:length(PFfiles)
    plot(R(i).Freq, abs(R(i).AccelFFT), LineWidth=1.5)
end
ylabel('Base')
xlim([0 7500]);

%% Functions

function rfft = computeFFT(input, N)
input_w_full = fft(input)/N;

input_w = input_w_full(1:N/2+1,:);
input_w(2:end-1,:) = 2*input_w(2:end-1,:);

rfft = input_w;
end
