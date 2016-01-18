%% Setup
clc
clear
close all

addpath(genpath('external'));

%% Propagation

run('parameters');
fpath = [data_path file_names{dbids}{file_id} '.mat'];
disp(fpath);    

load(fpath);
num_img=size(sequence_data,4);
r = size(sequence_data,1);
c = size(sequence_data,2);

%image_sequence strucutre contains all the raw data
image_sequence.sequence_data = sequence_data;
image_sequence.sequence_labels = sequence_labels;
image_sequence.forward_flow = forward_flow;
image_sequence.backward_flow = backward_flow;        
if(exist('superpixel_data','var'))
    image_sequence.superpixel_data = superpixel_data;
end

%Use motion term while computing error (1/0)
options.use_motion = 1;
%Use discrepancy term while computing error (1/0)
options.use_discrepancy = 1;
%Use color term while computing error (1/0)
options.use_color = 1;
%Displays the cost while computing if 1.
options.show_cost_plot = 0;

% num_img * num_img matrix which needs to be computed
% to make dynamic programming selection
cost_matrix = zeros(num_img,num_img);

% Compute the cost matrix which will be used for DP selection
for ind=1:num_img   
    tic;
    fprintf('Computing costs for frame number:%d  ',ind);
    [cumulative_cost] = compute_flow_error(image_sequence,ind, options);    
    cost_matrix(ind,:) = cumulative_cost;    
    toc;
end

image_sequence.cost_matrix = cost_matrix;
%Flow based propgation also needs to compute costs, hence need to pass 
%the cost_options to the propagation function
image_sequence.cost_options = options;
clear options;


%%

%%
%options.k : Number of frames to be selected

%options.propagation_method : Technique used for Label Propagation
%   -forward : Propagate labels only in forward direction
%   -nearest_neighbor : Propagate labels from the nearest neighbor in either
%                       direction
%   -pixelflow (ours) : Propagate labels using the pixelflow method
%   -mrf (ours) : Propagate labels using the MRF method

%options.selectoin_method : Technique used for Frame Selection
%   -uniform : Select frames at uniform intervals
%   -dp (ours) : Dynamic Programming based selection as described in paper

results = cell(length(numsel_frames),4);
options.show_plots = 0;             
for nind = 1:length(numsel_frames)
    numsel = numsel_frames(nind);
    cnt = 1;
    options.k = numsel;       
    prop_methods = {'forward','nearest_neighbor'};
    options.selection_method = 'uniform';                                         
    for t=1:size(prop_methods,2)
        options.propagation_method = prop_methods{t};
        disp([ options.selection_method ' ' prop_methods{t} '....']);
        [pixel_errors, avg_error, predicted_labels] = ...
            active_frame_selection(image_sequence,options);                                    
        disp([' k=' num2str(options.k) ' | Avg Error: ' num2str(avg_error)]);                
        result.k = options.k;
        result.propagation_method = options.propagation_method;
        result.selection_method = options.selection_method;
        result.pixel_errors = pixel_errors;
        result.avg_error = avg_error;
        result.predicted_labels = predicted_labels;
        results{nind,cnt} = result;
        cnt = cnt+1;
        clear result;
        disp('');
    end

    prop_methods = {'pixelflow','mrf'};
    options.selection_method = 'dp';                                            

    for t=1:size(prop_methods,2)        
        options.propagation_method = prop_methods{t};
        disp([ options.selection_method ' ' prop_methods{t} '....']);
        [pixel_errors, avg_error, predicted_labels] = ...
            active_frame_selection(image_sequence,options);                         
        disp([' k=' num2str(options.k) ' | Avg Error: ' num2str(avg_error)]);
        result.k = options.k;
        result.propagation_method = options.propagation_method;
        result.selection_method = options.selection_method;
        result.pixel_errors = pixel_errors;
        result.avg_error = avg_error;
        result.predicted_labels = predicted_labels;
        results{nind,cnt} = result;
        cnt = cnt+1;
        clear result;
    end  
    
    disp('=======================================');
end