clear
close all
clc
%% added velocity calculation 2/27/2024
% 
% DataFolder = 'C:\Users\1464465614E\Documents\Projects\JEMTP-MicrobeadedEncapsulants\DropTowerTesting\SNLSeries1\E5'; % choose unit to read

DataFolder = 'E:\415user_workspace\INTERN_2024\2025 Testing\Test Set 1\m116_p5KG\'; % choose unit to read


MainFolder = cd;
cd(DataFolder) % step into folder



PFfiles0 = dir('t0*.txt');
PFfiles1 = dir('t1*.txt');
PFfiles2 = dir('t2*.txt');
PFfiles3 = dir('t3*.txt');
PFfiles4 = dir('t4*.txt');
PFfiles5 = dir('t5*.txt');
PFfiles6 = dir('t6*.txt');

PFfiles = [PFfiles0;PFfiles1;PFfiles2;PFfiles3;PFfiles4;PFfiles5;PFfiles6];


numfiles = size(PFfiles,1);
testnum = numfiles; %%%%CHANGE if you want to see a test that is not the most recent
% testnum = 2;
[~,index] = sortrows({PFfiles.datenum}.'); PFfiles = PFfiles(index); clear index %sort files by date to match order of tests recorded in testnotes
% AccelFSIN = input('Enter Acceleration FSIN = ');
AccelFSIN=30000;
FSOUT = 10;

%% look at most recent test (end of PFfiles sorted data)


% PFdata= readmatrix(PFfiles(numfiles).name,'NumHeaderLines',8);
PFdata = dlmread(PFfiles(testnum).name,'\t',8,0); %read file skipping header see example in MATLAB folder
time = PFdata(:,end);
Accel = PFdata(:,1).*AccelFSIN./FSOUT;

% figure(100); 
% plot(time*1000,Accel)
% xlabel('Time [ms]')
% ylabel('Acceleration [G]')


%% get peak and duration
[peak, peakIdx] = max(Accel);
WidthPercent = 10;
w = pulsewidth(Accel,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peak]);
figure(10);
pulsewidth(Accel,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peak]);
xlim([time(peakIdx)-1e-3 time(peakIdx)+1e-3])
set(gcf,'Position',[1010 914 560 420])
disp(['Peak = ',num2str(peak),' G'])
disp(['PulseWidth = ',num2str(w*1000),'ms'])

str1 = {['Peak = ',num2str(peak),' G'],['PulseWidth = ',num2str(w*1000),'ms']};
annotation('textbox',[0.15,0.12,0.1,0.1],'String',str1,'FitBoxToText','on','BackgroundColor','w','FontWeight','bold');

% % k_pulseDetails(time,Accel)

%% filter and get peak and duration

fc = 50e3;
Fs = 1./mean(diff(PFdata(:,end)));
filterOrder = 2;
Wn = fc/(Fs/2);       % Normalized cutoff frequency = freq/(sampling freq/2)
[b,a] = butter(filterOrder,Wn,'low');           % IIR filter design
AccelFiltered = filtfilt(b,a,Accel);% zero-phase filtering

% figure(100)
% hold on
% plot(time*1000,AccelFiltered)
% legend('Raw','Filtered')
% xlim([time(peakIdx)-1e-3 time(peakIdx)+1e-3]*1000)

[peakF, peakIdxF] = max(AccelFiltered);
WidthPercent = 10;
[wF,initcross, finalcross,midlevel] = pulsewidth(AccelFiltered,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peakF]);
figure(11);
pulsewidth(AccelFiltered,time,'MidPercentReferenceLevel',WidthPercent,'StateLevels',[0 peakF]);
xlim([time(peakIdxF)-1e-3 time(peakIdxF)+1e-3])

disp(['Peak = ',num2str(peakF),' G'])
disp(['PulseWidth = ',num2str(wF*1000),'ms'])

str1 = {['Peak = ',num2str(peakF),' G'],['PulseWidth = ',num2str(wF*1000),'ms']};
annotation('textbox',[0.15,0.12,0.1,0.1],'String',str1,'FitBoxToText','on','BackgroundColor','w','FontWeight','bold');

%% calculate velocity
%get just pulse area
IdxEnd = find((Accel(peakIdx:end)<0),1)+peakIdx;

vel_vector = cumtrapz(time(1:IdxEnd),Accel(1:IdxEnd)*32.2); %[ft/s]
vel = trapz(time(1:IdxEnd),Accel(1:IdxEnd)*32.2); %[ft/s]
disp(['VelChange = ',num2str(vel),' ft/s'])
%% calculate fft of base accel

% input = Accel;
% if size(PFdata,2)<=2
%     response = NaN(size(PFdata(:,2))); %if no board accel, don't calculate a response fft
% else
%     response = PFdata(:,5).*AccelFSIN./FSOUT;;
% end
% T = 1/Fs; %sampling period
% N = length(time); %length of signal
% t = (0:N-1)*T; %time vector %SHOULD MATCH EXPERIMENTAL time vector if Fs is correct
% 
% freq = Fs*(0:N/2)/N; %define frequency vector [Hz]
% input_w_full = fft(input)/N;
% response_w_full = fft(response)/N;
% 
% input_w = input_w_full(1:N/2+1,:);
% input_w(2:end-1,:) = 2*input_w(2:end-1,:);
% 
% response_w = response_w_full(1:N/2+1,:);
% response_w(2:end-1,:) = 2*response_w(2:end-1,:);
% 
% responseFFT = response_w;
% inputFFT = input_w;
% dataFRF = response_w./input_w; % frequency resonse function response(SUT)/input(reference)

% figure('Position',[896 130 560 920]); 
% ax1 = subplot(3,1,1);
% plot(freq, abs(inputFFT))
% ylabel('FFT of Base Accel')
% xlabel('Frequency [Hz]')
% ax2 = subplot(3,1,2);
% plot(freq, abs(responseFFT))
% ylabel('FFT of Board Accel')
% xlabel('Frequency [Hz]')
% ax3 = subplot(3,1,3);
% plot(freq, abs(dataFRF))
% ylabel('Ratio')
% xlabel('Frequency [Hz]')
% linkaxes([ax1,ax2,ax3],'x')
% xlim([0 fc])
% 



%%
cd(MainFolder)
