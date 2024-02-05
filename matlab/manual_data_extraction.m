%% Configuration

clear all
close all
clc

files = ["/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5483/5483.220523170937.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6464/6464.220523163331.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6466/6466.220523162032.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6474/6474.220523174654.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5816/5816.220523180308.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5814/5814.220524023218.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6469/6469.220524021627.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5815/5815.220523153635.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6468/6468.220523155511.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6470/6470.220523184339.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6476/6476.220523182736.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6465/6465.220524014109.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5783/5783.220524005201.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6473/6473.220523172921.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6477/6477.220523165055.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6472/6472.220524030528.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5818/5818.220523190912.wav"];
timestamps = [35883+12, 
              38049+19,
              38828+13,
              33646+17,
              32672+17,
              2122+14,
              3073+14, 
              41465+12,
              40349+14,
              30241+20,
              31204+19,
              5191+17,
              8139+21,
              34699+19,
              37005+13,
              132+13,
              28708+8];
location = [40.44773701, -70.547042];

% settings
fs_save = 600; % [Hz]
T = 6; % [s]
save_index = 53;

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
h5create(file_name, "/sensors", size(sensorIDs),'Datatype','single')
h5create(file_name, "/location", size(location),'Datatype','single')
h5write(file_name, "/data", calls);
h5write(file_name, "/sensors", sensorIDs);
h5write(file_name, "/location", location);