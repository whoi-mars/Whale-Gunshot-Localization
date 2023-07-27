% paths to rsk file
rskFileDeploy = '/media/markgoldwater/Extreme SSD/CCB_2023/deployment_metadata/203246_20230404_1741.rsk';
rskFileRecover = '/media/markgoldwater/Extreme SSD/CCB_2023/recovery_metadata/203246_20230428_1824.rsk';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Deployment               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% deployment data
rskD = RSKopen(rskFileDeploy);
rskD = RSKreaddata(rskD);

% determine if CTD data is present
channels = {rskD.channels.longName};
if not(any(strcmp(channels, 'Temperature')) && ...
       any(strcmp(channels, 'Pressure')) && ... 
       any(strcmp(channels, 'Conductivity')))
  error("rsk file must include 'Temperature', 'Pressure' and 'Conductivity' channels.")
end

rskD = RSKreadprofiles(rskD);

% derive necessary data
rskD = RSKderiveseapressure(rskD);
rskD = RSKderivesalinity(rskD);
rskD = RSKderivedepth(rskD);
rskD = RSKderivesoundspeed(rskD);

figure;
[handleD, axesD] = RSKplotprofiles(rskD,'channel',{'Speed Of Sound','Temperature','Salinity'},'direction','down');
xlabel(axesD(2),'^{\circ}C')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Recovery                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% deployment data
rskR = RSKopen(rskFileRecover);
rskR = RSKreaddata(rskR);

% determine if CTD data is present
channels = {rskR.channels.longName};
if not(any(strcmp(channels, 'Temperature')) && ...
       any(strcmp(channels, 'Pressure')) && ... 
       any(strcmp(channels, 'Conductivity')))
  error("rsk file must include 'Temperature', 'Pressure' and 'Conductivity' channels.")
end

rskR = RSKreadprofiles(rskR);

% derive necessary data
rskR = RSKderiveseapressure(rskR);
rskR = RSKderivesalinity(rskR);
rskR = RSKderivedepth(rskR);
rskR = RSKderivesoundspeed(rskR);

figure;
[handleR, axesR] = RSKplotprofiles(rskR,'channel',{'Speed Of Sound','Temperature','Salinity'},'direction','down');
xlabel(axesR(2),'^{\circ}C')