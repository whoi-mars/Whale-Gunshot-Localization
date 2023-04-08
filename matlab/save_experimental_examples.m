%% CLEAR EVERYTHING

clear
clc
close all

%% CONFIG

% table with experimental example information
T = readtable('ccb_2022.csv');

% file path to save experimental data example
file_path = 'ccb_2020_exp_data.h5';

% control how much to the left and right of timestamp is saved
window_delta = 80; % [sec]

% example length
example_length = 6; % [sec]

% sample rate of data
fs = 24000; % [Hz]

% desired sample rate to save
fs_desired = 600; % [Hz]

% row number
row = 1;

% take L2 norm when saving?
L2NORM = 1;

% simulated order of TOSSITs
sim_order = ["6474", "6470", "6471", "6476", "6468"];

% csv order
csv_order = ["6470", "6468", "6471", "6474", "6476"];

%-------------------------------------------------------------------------%

% Number of TOSSITs
num_TOSSITs = (size(T, 2) - 2) / 3;

% number of rows
num_rows = size(T, 1);

% create reindexing list to save data in the same order in which it was
% simulated
reidx = zeros(1,length(csv_order));
for i = 1:length(reidx)
    reidx(i) = find(csv_order == sim_order(i));
end

%% WIDE VIEW OF TABLE ROW

% collect mean-centered signals
signals = [];
for i = 1:num_TOSSITs
    [y, fs_file] = audioread(T(row, "File" + string(i)).(1){1}, [round(T(row, "Start" + string(i)).(1)*fs), round((T(row, "Start" + string(i)).(1) + window_delta*2)*fs)]);
    y = resample(y, fs_desired, fs);
    signals = [signals; (y - mean(y)).'];
end

% spectrogram parameters
Nw = 45;
noverlap = Nw-8;
nfft = 1012;

% plot spectrograms
figure;
for c = 1:num_TOSSITs
    subplot(5,1,c)
    [stft,f,t] = spectrogram(signals(c,:),Nw,noverlap,nfft,fs_desired);
    imagesc(t,f,10*log10(abs(stft).^2))
    colormap('turbo')
    axis xy
    xlabel("Time [s]")
    ylabel("Frequency [Hz]")
end

%% CALL VIEW OF TABLE ROW

% collect shorter windows which will be turned into saved examples
signals = [];
for i = 1:num_TOSSITs
    start_sample = round((T(row, "Start" + string(i)).(1) + T(row, "CallStart" + string(i)).(1))*fs); 
    [y, fs_file] = audioread(T(row, "File" + string(i)).(1){1}, [start_sample, start_sample + example_length*fs - 1]);
    y = resample(y, fs_desired, fs);
    signals = [signals; (y - mean(y)).'];
end

% spectrogram parameters
Nw = 45;
noverlap = Nw-8;
nfft = 1012;

% plot spectrograms
figure
for c = 1:num_TOSSITs
    subplot(5,1,c)
    [stft,f,t] = spectrogram(signals(c,:),Nw,noverlap,nfft,fs_desired);
    imagesc(t,f,10*log10(abs(stft).^2))
    colormap('turbo')
    axis xy
    xlabel("Time [s]")
    ylabel("Frequency [Hz]")
end

%% SAVE SIGNALS

% create matrix for saved signals
signals = zeros(fs_desired*example_length, num_TOSSITs, num_rows, 'single');
labels = zeros(2, num_rows);

% collect signals to save, reindexing so that the TOSSITs are in the same
% order which was used for simulation
for j = 1:num_rows
    for i = 1:num_TOSSITs
        i = reidx(i);
        start_sample = round((T(j,"Start" + string(i)).(1) + T(j,"CallStart" + string(i)).(1))*fs); 
        [y, fs_file] = audioread(T(row, "File" + string(i)).(1){1}, [start_sample, start_sample + example_length*fs - 1]);
        y = resample(y, fs_desired, fs);
        y = (y - mean(y)).';
        
        if L2NORM
            y = y ./ sqrt(sum(y.^2));
        end
        
        signals(:,i,num_rows) = y;
    end
    
    % save labels from csv
    labels(1,j) = T(j, "Yloc").(1);
    labels(2,j) = T(j, "Xloc").(1);
end

% get chunk size
if num_rows > 1
    data_chunk_size = [size(signals,[1 2]) 1];
    label_chunk_size = [2];
else
    data_chunk_size = [size(signals,[1 2])];
    label_chunk_size = [2 1];
end

% save data
h5create(file_path, "/data", size(signals), 'Datatype', 'single', 'ChunkSize', data_chunk_size);
h5create(file_path, "/labels", [2 num_rows], 'Datatype', 'double', 'ChunkSize', label_chunk_size);
h5write(file_path, "/data", signals);
h5write(file_path, "/labels", labels); 