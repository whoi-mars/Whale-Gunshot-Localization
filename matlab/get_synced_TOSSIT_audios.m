%% CLEAR EVERYTHING

clear
clc
close all

%% CONFIG

% root containing folder with data from each TOSSIT
base_dir = "/media/markgoldwater/Elements/cape_cod_bay/acoustics/";
% location of source wav file where a signal of interest was identified
% source_file = fullfile(base_dir, "6470/6470.220407001558.wav");
% % timestamp in the source file
% timestamp = 37955; % [sec]
source_file = fullfile(base_dir, "6470/6470.220407001558.wav");
% timestamp in the source file
timestamp = 37995; % [sec]
% control how much to the left and right of timestamp is saved
window_delta = 80; % [sec]
window_delta_source = 80; % [sec]
% sample rate of data
fs = 24000; % [Hz]
% desired sample rate to save
fs_desired = 600; % [Hz]

%% GET SIGNALS

% get TOSSIT ID and file number
split_path = split(source_file,["/","."]);
TOSSIT_id = split_path(end-2);
source_file_num = split_path(end-1);

% get sample from source file
[y, fs_file] = audioread(source_file,[fs*(timestamp - window_delta_source), fs*(timestamp + window_delta_source)]);
assert(fs == fs_file, "sampling frequency from file does not match expected");
result.source = resample(y, fs_desired, fs);
result.fs = fs_desired;
clear y fs_file

% get start time of source file
xml_source = readstruct(fullfile(base_dir,TOSSIT_id,TOSSIT_id + "." + source_file_num + ".log.xml"));
start_time_struct_source = xml_source.PROC_EVENT(2).WavFileHandler;
assert(isfield(start_time_struct_source,'SamplingStartTimeLocalAttribute'),...
       "XML file does not contain proper WavFileHandler filed");
start_time_source = start_time_struct_source.SamplingStartTimeLocalAttribute;

% get root directory info
baseInfo = dir(base_dir);
isub = [baseInfo(:).isdir];
TOSSIT_id_list = {baseInfo(isub).name};
TOSSIT_id_list(ismember(TOSSIT_id_list,{'.','..',char(TOSSIT_id)})) = [];

closest_time_list = NaT(1,length(TOSSIT_id_list));
closest_xml_list = cell(1,length(TOSSIT_id_list));
for i = 1:length(TOSSIT_id_list)
    % store current TOSSIT
    curr_TOSSIT_id = TOSSIT_id_list(i);
    
    % find closest XML file and corresponding start time to source file
    [closest_xml_file, closest_start_time] = getClosestStartTimeFile(fullfile(base_dir,curr_TOSSIT_id),start_time_source);
    
    % store
    closest_xml_list(i) = closest_xml_file;
    closest_time_list(i) = closest_start_time;
end

% retrieve corresponding audio clip from each TOSSIT adjusted for clocks
% not being synchronized
for j = 1:length(closest_xml_list)
    
    % determine offset for this TOSSIT
    delta = start_time_source - closest_time_list(j);
    
    % warn if delta is very large
    if delta > duration(0,5,0)
        warning("TOSSIT time difference is larger than 5 minutes");
    end
    
    % construct audio file path for current TOSSIT
    split_xml = split(closest_xml_list(j),".");
    wav_path = fullfile(base_dir,split_xml(1),split_xml(1) + "." + split_xml(2) + ".wav");
    
    % load audio snippet from TOSSIT which correspnds to source, accounting
    % for not being synchronized
    [y, fs_file] = audioread(wav_path,[fs*(timestamp + seconds(delta) - window_delta), fs*(timestamp + seconds(delta) + window_delta)]);
    assert(fs == fs_file, "sampling frequency from file does not match expected");
    result.("TOSSIT" + split_xml(1)) = resample(y, fs_desired, fs);
    clear y fs_file
end

%% PLOT RAW RESULTS

% plot spectrograms of results
Nw = 45;
noverlap = Nw-8;
nfft = 1012;

figure;
[stft,f,t] = spectrogram(result.source,Nw,noverlap,nfft,result.fs);
subplot(5,1,1)
imagesc(t,f,10*log10(abs(stft).^2))
colormap('turbo')
axis xy
xlabel("Time [s]")
ylabel("Frequency [Hz]")
title("TOSSIT " + TOSSIT_id + " (source)")

for c = 1:length(TOSSIT_id_list)
    subplot(5,1,c+1)
    [stft,f,t] = spectrogram(result.("TOSSIT" + TOSSIT_id_list(c)),Nw,noverlap,nfft,result.fs);
    imagesc(t,f,10*log10(abs(stft).^2))
    colormap('turbo')
    axis xy
    xlabel("Time [s]")
    ylabel("Frequency [Hz]")
    title("TOSSIT " + TOSSIT_id_list(c))
end

sgt = sgtitle("Rough Synchronize");
sgt.FontSize = 30;

%% PLOT SYNCHRONIZED RESULTS

result.deltas = zeros(1,length(TOSSIT_id_list));
for t = 1:length(TOSSIT_id_list)
    [sig,I] = spectrogramCorr(result.source, result.("TOSSIT" + TOSSIT_id_list(t)), Nw, noverlap, nfft, result.fs);
    result.("TOSSIT" + TOSSIT_id_list(t) + "_synced") = sig;
    result.deltas(t) = I;
end

% plot spectrograms of results
Nw = 45;
noverlap = Nw-8;
nfft = 1012;

figure;
[stft,f,t] = spectrogram(result.source,Nw,noverlap,nfft,result.fs);
subplot(5,1,1)
imagesc(t,f,10*log10(abs(stft).^2))
colormap('turbo')
axis xy
xlabel("Time [s]")
ylabel("Frequency [Hz]")
title("TOSSIT " + TOSSIT_id + " (source)")

for c = 1:length(TOSSIT_id_list)
    subplot(5,1,c+1)
    [stft,f,t] = spectrogram(result.("TOSSIT" + TOSSIT_id_list(c) + "_synced"),Nw,noverlap,nfft,result.fs);
    imagesc(t,f,10*log10(abs(stft).^2))
    colormap('turbo')
    axis xy
    xlabel("Time [s]")
    ylabel("Frequency [Hz]")
    title("TOSSIT " + TOSSIT_id_list(c))
end
sgt = sgtitle("Fine Synchronize");
sgt.FontSize = 30;

%% FUNCTIONS

function [xml_file, closest_start_time] = getClosestStartTimeFile(xmls_path, start_time_source)
    
    % get all XML file names
    xmls = dir(xmls_path + "/*.xml");
    xmls = {xmls.name};
    
    start_times = NaT(1,length(xmls));
    for x = 1:length(xmls)
        % get struct containing start time and sanity check its presence
        xml = readstruct(fullfile(xmls_path,xmls(x)));
        start_time_struct = xml.PROC_EVENT(2).WavFileHandler;
        assert(isfield(start_time_struct,'SamplingStartTimeLocalAttribute'),...
               "XML file does not contain proper WavFileHandler filed"); 
        
        % save start time
        start_times(x) = start_time_struct.SamplingStartTimeLocalAttribute;
    end
    
    % find index of closest start time
    [~,ind_closest] = min(abs(start_times - start_time_source));
    
    % assign return variables
    closest_start_time = start_times(ind_closest);
    xml_file = xmls(ind_closest);
end

function [sig_shifted, I] = spectrogramCorr(x, y, Nw, noverlap, nfft, fs)
    
    % calculate spectrograms
    [stftx,~,~] = spectrogram(x,Nw,noverlap,nfft,fs);
    [stfty,~,~] = spectrogram(y,Nw,noverlap,nfft,fs);
    
    % put on log scale
    stftx = 10*log10(abs(stftx).^2);
    stfty = 10*log10(abs(stfty).^2);
    
    % calculate channel-wise xcorrs
    y_dim_res = 2*max(size(stftx,2),size(stfty,2)) - 1;
    x_dim_res = size(stftx,1);
    cc = zeros(x_dim_res, y_dim_res);
    for i=1:x_dim_res
        [c, lags] = xcorr(stftx(i,:) - mean(stftx(i,:)), stfty(i,:) - mean(stfty(i,:)), 'coeff');
        cc(i,:) = c;
    end
    
    % sum each channel and get max xcorr
    cc = sum(cc,1);
    [~,I] = max(cc);
    
    % convert shift to signal samples rather than spectrogram samples
    sig_spect_sample_ratio = round(length(x) / size(stftx,2));
    I = sig_spect_sample_ratio*lags(I);
    
    % rotate signal
    sig_shifted = circshift(y, I);
    if I > 0
        sig_shifted(1:I) = 0;
    else
        sig_shifted(I:end) = 0;
    end
end