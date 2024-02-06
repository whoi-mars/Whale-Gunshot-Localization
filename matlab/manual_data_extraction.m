%% Configuration

clear all
close all
clc

files = ["/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5483/5483.220523170937.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6464/6464.220523163331.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6466/6466.220523162032.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6474/6474.220523174654.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5816/5816.220523180308.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5814/5814.220523143227.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6469/6469.220523141634.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5815/5815.220523153635.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6468/6468.220523155511.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6470/6470.220523184339.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6476/6476.220523182736.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6465/6465.220524014109.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5783/5783.220524005201.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6473/6473.220523172921.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6477/6477.220523165055.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/6472/6472.220523150538.wav"
         "/media/mark/extradrive2/SBCEX22/acoustics/2_circle_tow_riuss/5818/5818.220523190912.wav"];
timestamps = [31373+6-10,
              33539+16-10,
              34318+10-10,
              29136+11-10,
              28162+10-10,
              40803+14-10,
              41756+11-10,
              36955+8-10,
              35839+10-10,
              25731+14-10,
              26694+13-10,
              681+11-10,
              3629+15-10,
              30189+13-10,
              32495+10-10,
              38812+10-10,
              24198+2-10];
location = [40.46041598, -70.56697802];

% settings
fs_save = 600; % [Hz]
T = 20; % [s]
save_index = 49;

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
file_name = sprintf(fullfile(root, "gunshots_long_%d.h5"), save_index);
h5create(file_name, "/data", size(calls), 'Datatype','single');
h5create(file_name, "/sensors", size(sensorIDs),'Datatype','single');
h5create(file_name, "/location", size(location),'Datatype','single');
h5create(file_name, "/timestamps", size(timestamps),'Datatype','single');
h5create(file_name, "/files", size(files), 'Datatype', 'string');
h5write(file_name, "/data", calls);
h5write(file_name, "/sensors", sensorIDs);
h5write(file_name, "/location", location);
h5write(file_name, "/timestamps", single(timestamps));
h5write(file_name, "/files", files);