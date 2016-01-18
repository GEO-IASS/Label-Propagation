clear all;
warning off;
addpath('../external/netlab');
addpath('../external/GraphCut');


file_names{1} = {'cheetah','girl','birdfall2','penguin',...
    'monkeydog','parachute'}; %SegTrack Data
file_names{2} = {'CamSeq01'}; %CamSeq Data
file_names{3} =  {'LabelMeVideo'}; %LabelMe data

data_path = '../data/'; %path where the actual data is

numsel_frames = [5 10]; %Number of frames to be annotated
dbids = 1; file_id = 6; %This means use girl sequence from segtrack data
