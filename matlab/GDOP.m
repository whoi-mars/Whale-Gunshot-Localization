%% Setup

clear all
close all
clc

% define bounds of grid
max_x = 36.153303773;
min_x = -18.318595602;
min_y = -17.420482956;
max_y = 20.107685114;

% define sensors locations
all_sensors = [0 0     
               0 5003.756  
               0 10007.51   
               5026.815 10007.51   
               10260.21 10007.51   
               15011.58 10007.51   
               15022.2 5003.756  
               15376.18 648.635  
               10067.71 92.66214
               9991.828 5003.756  
               4961.394 5096.418  
               5033.853 92.66214] / 1000;

% finds the closest k - 1 sensors to each sensor
k = 5;

% dimensions of subplot (prod(subplot_dims) must equal size(all_sensors,1))
subplot_dims = [3 4];

%% Get chunks of n sensors

if k == size(all_sensors,1)
    sensor_groups = reshape(all_sensors,horzcat(1,size(all_sensors)));
else
    sensor_groups = zeros(12,k,2);
    for s = 1:size(all_sensors,1)
        [~,I] = mink(sqrt(sum((all_sensors - all_sensors(s,:)).^2,2)),k);
        sensor_groups(s,:,:) = all_sensors(I,:);
    end
end

%% GDOP

if k == size(all_sensors,1)
    assert(isequal(subplot_dims,[1 1]), "subplot dimensions do not work for the number of plots")
else
    assert(prod(subplot_dims) == size(all_sensors,1), "subplot dimensions do not work for the number of plots")
end

% rail the bondaries of the grid to integers
max_x = ceil(max_x);
min_x = floor(min_x);
max_y = ceil(max_y);
min_y = floor(min_y);

% get number of sensors
num_sensors = size(sensor_groups,2);

% create meshgrid
x_axis = min_x:0.1:max_x;
y_axis = min_y:0.1:max_y;
[X,Y] = meshgrid(x_axis,y_axis);

% initialize GDOP
GDOP = zeros(horzcat(size(sensor_groups,1),size(X)));

% initialize waitbar
f = waitbar(0);

for s = 1:size(sensor_groups,1)
    % update waitbar
    waitbar(s/size(sensor_groups,1),f,"Calculating GDOP...");
    
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
            GDOP(s,i,j) = sqrt(trace(Q));
        end
    end
end
close(f);

%% Make Plots

if k == size(all_sensors,1)
    imagesc(x_axis, y_axis, squeeze(GDOP(1,:,:)))
    axis xy
    hold on
    plot(squeeze(sensor_groups(1,:,1)),squeeze(sensor_groups(1,:,2)),'yx','MarkerSize',10)
    title("Geometric Dilution of Precision") 
    colorbar
else
    % set limits of colorbar
    minColorLimit = min(GDOP,[],'all');                   
    maxColorLimit = max(GDOP,[],'all');
    
    % make subplots
    fig = figure(1);
    for s = 1:size(all_sensors,1)
        sph = subplot(subplot_dims(1),subplot_dims(2),s,'Parent',fig);
        imagesc(x_axis, y_axis, squeeze(GDOP(s,:,:)))
        caxis(sph,[minColorLimit,maxColorLimit]);
        axis xy
        hold on
        plot(squeeze(sensor_groups(s,:,1)),squeeze(sensor_groups(s,:,2)),'yx','MarkerSize',10)
    end
    
    % set axis details
    h = axes(fig,'visible','off'); 
    h.Title.Visible = 'on';
    h.XLabel.Visible = 'on';
    h.YLabel.Visible = 'on';
    ylabel(h,'Y [km]','FontWeight','bold');
    xlabel(h,'X [km]','FontWeight','bold');
    title(h,'Geometric Dilution of Precision','FontSize',16);
    
    % set colorbar
    c = colorbar(h,'Position',[0.93 0.168 0.022 0.7]);
    colormap(c,'jet')
    caxis(h,[minColorLimit,maxColorLimit]);
end