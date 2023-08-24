%% Config

clear all
close all
clc

% define bounds of grid
x_bounds = [-18.318595601991970, 36.153303772987925];
y_bounds = [-17.420482956169757, 20.107685114302324];

sensor_latlons = [41.910825, -70.429297;
                  41.956023, -70.429105;
                  42.001120, -70.428992;
                  42.001098, -70.368635;
                  42.000940, -70.305142;
                  42.001002, -70.247648;
                  41.955833, -70.247817;
                  41.916856, -70.243174;
                  41.911469, -70.307519;
                  41.955866, -70.308182;
                  41.957085, -70.368867;
                  41.911413, -70.368597];

% define sensors locations
% all_sensors = [0 0     
%                0 5003.756  
%                0 10007.51   
%                5026.815 10007.51   
%                10260.21 10007.51   
%                15011.58 10007.51   
%                15022.2 5003.756  
%                15376.18 648.635  
%                10067.71 92.66214
%                9991.828 5003.756  
%                4961.394 5096.418  
%                5033.853 92.66214] / 1000;

% finds the closest k - 1 sensors to each sensor
k = 3;

% dimensions of subplot (prod(subplot_dims) must equal size(all_sensors,1))
subplot_dims = [3 4];

% grid density
gd = 400;

%% Setup

% number of sensors
total_num_sensors = size(sensor_latlons,1);

% load MA shape/location
states = readgeotable("usastatehi.shp");
MA = states(states.Name == "Massachusetts",:);

% get edges of MA
us_states = shaperead("usastatehi.shp");
MA_Lons = us_states(21).X;
MA_Lats = us_states(21).Y;

% derive lat/lon limits
[lat,lon] = reckon(sensor_latlons(1,1),sensor_latlons(1,2),abs(km2deg(y_bounds(1))),180);
lat_bounds(1) = lat;
[lat,lon] = reckon(sensor_latlons(1,1),sensor_latlons(1,2),abs(km2deg(y_bounds(2))),0);
lat_bounds(2) = lat;
[lat,lon] = reckon(sensor_latlons(1,1),sensor_latlons(1,2),abs(km2deg(x_bounds(1))),-90);
lon_bounds(1) = lon;
[lat,lon] = reckon(sensor_latlons(1,1),sensor_latlons(1,2),abs(km2deg(x_bounds(2))),90);
lon_bounds(2) = lon;

% make vectors of lat/lon
lat_vec = linspace(lat_bounds(1),lat_bounds(2),gd);
lon_vec = linspace(lon_bounds(1),lon_bounds(2),gd);

% determine lat/lons which are in MA
[LON,LAT] = meshgrid(lon_vec,lat_vec);
in = flipud(inpolygon(LON,LAT,MA_Lons,MA_Lats));

% derivate meter locations of all sensors relative to the first
all_sensors = zeros(total_num_sensors,2);
for i=1:total_num_sensors
    [d,az] = distance(sensor_latlons(1,1),sensor_latlons(1,2),sensor_latlons(i,1),sensor_latlons(i,2));
    d = deg2km(d);
    x = d*cosd(az);
    y = d*sind(az);
    all_sensors(i,:) = [y x];
end

%% Get chunks of n sensors

if k == size(all_sensors,1)
    sensor_groups = reshape(all_sensors,horzcat(1,size(all_sensors)));
    sensor_groupsll = reshape(sensor_latlons,horzcat(1,size(sensor_latlons)));
else
    sensor_groups = zeros(total_num_sensors,k,2);
    sensor_groupsll = zeros(total_num_sensors,k,2);
    for s = 1:size(all_sensors,1)
        [~,I] = mink(sqrt(sum((all_sensors - all_sensors(s,:)).^2,2)),k);
        sensor_groups(s,:,:) = all_sensors(I,:);
        sensor_groupsll(s,:,:) = sensor_latlons(I,:);
    end
end

%% GDOP

if k == size(all_sensors,1)
    assert(isequal(subplot_dims,[1 1]), "subplot dimensions do not work for the number of plots")
else
    assert(prod(subplot_dims) == size(all_sensors,1), "subplot dimensions do not work for the number of plots")
end

% get number of sensors
num_sensors = size(sensor_groups,2);

% create meshgrid
x_axis = linspace(x_bounds(1), x_bounds(2),gd);
y_axis = linspace(y_bounds(1),y_bounds(2),gd);
[X,Y] = meshgrid(x_axis,y_axis);

% initialize GDOP
GDOP = zeros(horzcat(size(sensor_groups,1),size(X)));

% initialize waitbar
f = waitbar(0);

for s = 1:size(sensor_groups,1)
    % update waitbar
    waitbar(s/size(sensor_groups,1),f,"Calculating GDOPs...");
    
    % get relevant sensors
    sensors = squeeze(sensor_groups(s,:,:));

    % add third dimension (one for each sensor)
    X_3d = repmat(X,1,1,size(sensors,1));
    Y_3d = repmat(Y,1,1,size(sensors,1));

    % calculate distances between sample points and sensors
    R = sqrt((reshape(sensors(:,1),[1 1 num_sensors]) - X_3d).^2 + (reshape(sensors(:,2),[1 1 num_sensors]) - Y_3d).^2);
    
    % calulate partial derivatives of distance formula
    X_3d_div = (reshape(sensors(:,1),[1 1 num_sensors]) - X_3d) ./ R;
    Y_3d_div = (reshape(sensors(:,2),[1 1 num_sensors]) - Y_3d) ./ R;

    for i=1:size(X,1)
        for j=1:size(X,2)
            A = squeeze([X_3d_div(i,j,:) Y_3d_div(i,j,:)]).';
            Q = 1*inv(A'*A);
            GDOP(s,i,j) = sqrt(trace(Q) - Q(end, end));
        end
    end
end
close(f);

%% Make Plots

if k == size(all_sensors,1)
    % create USA map around MA
    ax = usamap(lat_bounds,lon_bounds);
    setm(ax,"FontSize",13)
    
    % display GDOP data
    pcolorm(lat_vec,...
            lon_vec,...
            squeeze(GDOP(1,:,:)));
    
    % plot MA shape and sensors
    geoshow(MA)
    geoshow(sensor_latlons(:,1),sensor_latlons(:,2), 'DisplayType', 'MultiPoint', 'Marker', 'o','MarkerFaceColor','yellow', 'Color', 'green','MarkerSize',20);
    
    % set up colorbar so the max color does not consider GDOP values on
    % land
    GDOPf = flipud(squeeze(GDOP(s,:,:)));
    minColorLimit = min(GDOPf(in ~= 1),[],'all');                   
    maxColorLimit = max(GDOPf(in ~= 1),[],'all');
    colorbar;
    clim([minColorLimit,maxColorLimit]);
else
    % make subplots
    fig = figure(1);
    for s = 1:size(all_sensors,1)
        % add to subplot
        sph = subplot(subplot_dims(1),subplot_dims(2),s,'Parent',fig);

        % create USA map around MA
        ax = usamap(lat_bounds,lon_bounds);
        setm(ax,"FontSize",13)
        
        % display GDOP data
        pcolorm(lat_vec,...
                lon_vec,...
                squeeze(GDOP(s,:,:)));
        
        % plot MA shape and sensors
        geoshow(MA)
        geoshow(sensor_groupsll(s,:,1),sensor_groupsll(s,:,2), 'DisplayType', 'MultiPoint', 'Marker', 'o','MarkerFaceColor','yellow', 'Color', 'green','MarkerSize',20);        
        
        % set up colorbar so the max color does not consider GDOP values on
        % land
        GDOPf = flipud(squeeze(GDOP(s,:,:)));
        minColorLimit = min(GDOPf(in ~= 1),[],'all');                   
        maxColorLimit = max(GDOPf(in ~= 1),[],'all');
        colorbar;
        clim([minColorLimit,maxColorLimit]);
    end
    
    % set up figures
    han=axes(fig,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    ylabel(han,'Latitude');
    xlabel(han,'Longitude');
    title(han,'Geometric Dilution of Precision');
    han.XLabel.Position(2) = -0.06;
    han.XLabel.FontSize = 24;
    han.YLabel.Position(1) = -0.06;
    han.YLabel.FontSize = 24;
    han.Title.FontSize = 24;
end