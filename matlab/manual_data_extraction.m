%% Configuration

clear all
close all
clc

% files and start times
files = ["/media/mark/extradrive3/CCB_2023/acoustics/5816/5816.230411131402.wav",...
         "/media/mark/extradrive3/CCB_2023/acoustics/5818/5818.230411124935.wav",...
         "/media/mark/extradrive3/CCB_2023/acoustics/6469/6469.230411043317.wav",...
         "/media/mark/extradrive3/CCB_2023/acoustics/6472/6472.230411045752.wav"];
timestamps = [5891,...
              7358,...
              37155,...
              35661]; % [s]

% settings
fs_save = 600; % [Hz]
T = 6; % [s]
save_index = 1;

%% Save Audio

assert(length(files) == length(timestamps),"files and timestamps must correspond.")

% get config information
addpath('yaml')
config = yaml.loadFile("../whale_gunshot_localization/config_12.yaml");

% get list of TOSSITs
TOSSIT_list = [];
for i = 1:length(config.TOSSIT.ids)
    TOSSIT_list = [TOSSIT_list double(config.TOSSIT.ids{i})];
end

% load and save sounds
for i = 1:length(files)
    split_path = split(files(i),'/');
    sensor_id = split_path{7};
    [in,idx] = ismember(str2num(sensor_id),TOSSIT_list);
    if ~in
        warning("no data for sensor %s.");
    else
        % get sample rate
        fs = audioinfo(files(i)).SampleRate;
        [y,fs] = audioread(files(i),[fs*timestamps(i), fs*(timestamps(i) + T ) - 1]);
        y = resample(y,fs_save,fs);
        audiowrite(sprintf("gunshot_%d_%d.wav",save_index,idx),y,fs_save);
    end
end