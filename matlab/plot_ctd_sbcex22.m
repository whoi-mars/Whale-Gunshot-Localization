%% Config
close all
clear all
clc

% range of CTD casts to plot
CTD=[10 33];

%% Make Plots
n = CTD(2) - CTD(1) + 1;
rows = floor(sqrt(n));
columns = ceil(n/rows);
f = figure;
tcl = tiledlayout(rows,columns);

% track min/max sound speed
max_ss = -inf;
min_ss = inf;

for i=CTD(1):CTD(2)
    % relevant files
    filename=['/media/markgoldwater/Extreme SSD/SBCEX22/data_env_conn/ctd_data/CTD' num2str(i) '.asc'];
    filename_hdr=['/media/markgoldwater/Extreme SSD/SBCEX22/data_env_conn/ctd_data/CTD' num2str(i) '.hdr'];
    
    % extract system date
    fid = fopen(filename_hdr);
    d = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',18);
    d = cell2mat(d{1});
    d = split(d,' = ');
    d = string(d{2});
    fclose(fid);
    
    % extract SSP
    fid = fopen(filename);
    tline = fgetl(fid);
    ii=1;
    while ischar(tline)
        tline = fgetl(fid);
        if tline ~=-1
            A(ii,:)=str2num(tline);
            ii=ii+1;
        end
    end
    fclose(fid);
    
    % filter
    toto=find(A(:,1)>2 & A(:,1)<200);
    A = A(toto,:);
    
    % update min/max sound speeds
    ss = max(A(:,end));
    if ss > max_ss
        max_ss = ss;
    end
    ss = min(A(:,end));
    if ss < min_ss
        min_ss = ss;
    end
    
    nexttile
    plot(A(:,end), A(:,1), 'linewidth',2)
    title(d)
    axis ij
end
title(tcl,'CTD Data')
xlabel(tcl,'Sound Speed [m/s]')
ylabel(tcl,'Depth [m]')
fontsize(f,0.2,"inches");