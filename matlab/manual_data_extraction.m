%% Configuration

clear all
close all
clc

files = ["/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5483/5483.220524050931.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6464/6464.220524043319.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6466/6466.220524042023.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6474/6474.220524054643.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5816/5816.220524060300.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5814/5814.220524023218.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6469/6469.220524021627.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5815/5815.220524033628.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6468/6468.220524035458.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6470/6470.220524064328.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6476/6476.220524062724.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6465/6465.220524014109.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5783/5783.220524005201.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6473/6473.220524052914.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6477/6477.220524045044.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6472/6472.220524030528.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5818/5818.220524070906.wav"];
timestamps = [8769+9
              10941+19
              11717+13
              6537+6  
              5560+4
              18202+12
              19153+13
              14352+10
              13242+12
              3132+10
              4096+7
              21271+11
              24219+15
              7586+12
              9896+13
              16212+12
              1594+2];
source_id = zeros(17,1,'single');
location = [40.517471, -70.70127304];

% settings
fs_save = 600; % [Hz]
T = 6; % [s]
save_index = 60;

%% Save Audio

assert(length(files) == length(timestamps),"files and timestamps must correspond.")

% get config information
addpath('yaml')
config = yaml.loadFile("../whale_gunshot_localization/config_mudpatch.yaml");

% get list of TOSSITs
TOSSIT_list = [];
for i = 1:length(config.TOSSIT.ids)
    TOSSIT_list = [TOSSIT_list double(config.TOSSIT.ids{i})];
end

% load calls
calls = zeros(T*fs_save,length(config.TOSSIT.ids),'single');
sensorIDs = zeros(length(files),1,'single');
location = single(location);
for i = 1:length(files)
    split_path = split(files(i),'/');
    sensor_id = split_path{8};
    [in,idx] = ismember(str2num(sensor_id),TOSSIT_list);
    if ~in
        warning("no data for sensor %s.");
    else
        % get sample rate
        fs = audioinfo(files(i)).SampleRate;
        [y,fs] = audioread(files(i),[fs*timestamps(i), fs*(timestamps(i) + T ) - 1]);
        y = resample(y,fs_save,fs);
        calls(:,i) = y;
        sensorIDs(i) = str2num(sensor_id);
    end
end

% save calls
root = split(config.dataset.ccb_data_directory,"/");
root = fullfile(strjoin(root(1:5),"/"), "single_examples");
file_name = sprintf(fullfile(root, "gunshots_%d.h5"), save_index);
h5create(file_name, "/data", size(calls), 'Datatype','single');
h5create(file_name, "/sensors", size(sensorIDs),'Datatype','single');
h5create(file_name, "/location", size(location),'Datatype','single');
h5create(file_name, "/timestamps", size(timestamps),'Datatype','single');
h5create(file_name, "/associations", size(timestamps.'),'Datatype','single')
h5create(file_name, "/files", size(files), 'Datatype', 'string');
h5write(file_name, "/data", calls);
h5write(file_name, "/sensors", sensorIDs);
h5write(file_name, "/location", location);
h5write(file_name, "/timestamps", single(timestamps));
h5write(file_name, "/files", files);
h5write(file_name, "/associations",zeros(1,length(timestamps),'single'))