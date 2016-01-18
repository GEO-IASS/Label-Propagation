function [pixel_errors,avg_error,predicted_labels] = ...
                    active_frame_selection(image_sequence,options)
   
num_img=size(image_sequence.sequence_data,4);

if(options.k<=0)
    disp('You need to choose atleast one frame');
    return;
end

num_selection = options.k;

if(strcmp(options.selection_method,'uniform')==1) 
    val2 = ceil(num_img/num_selection);
    val1 = floor(num_img/num_selection);
    if(val1 == val2)
        m = num_selection;
    else
        m = num_selection*val2 - num_img;
    end    
    selected_frames = [1:val1:m*val1 (m*val1+1):val2:num_img];
    if(length(selected_frames) == num_selection-1)
        selected_frames(end+1) = num_img;
    end       
elseif(strcmp(options.selection_method,'dp')==1)    
    selected_frames = dpsel(image_sequence.cost_matrix,num_selection)';
elseif(strcmp(options.selection_method,'fixed')==1)    
    selected_frames = options.selected_frames;
else
    fprintf('Need to specify a selection method\n');
    return;
end

if(strcmp(options.propagation_method,'forward')==1)        
    predicted_labels = forward_propagation(selected_frames, image_sequence);
elseif(strcmp(options.propagation_method,'nearest_neighbor')==1)    
    predicted_labels = nearest_neighbor_propagation(selected_frames, image_sequence);
elseif(strcmp(options.propagation_method,'pixelflow')==1)    
    predicted_labels = pixelflow_propagation(selected_frames, image_sequence);    
elseif(strcmp(options.propagation_method,'mrf')==1)    
    predicted_labels = mrf_propagation(selected_frames, image_sequence);    
else
    fprintf('Need to specify a propagation method\n');
    return;
end

pixel_errors = compute_pixel_errors(image_sequence, predicted_labels);
avg_error = mean(pixel_errors);

if(options.show_plots == 1)
    show_figures(image_sequence,predicted_labels);
end
end

function [predicted_labels] = forward_propagation(selected_frames, image_sequence)
    
    num_img=size(image_sequence.sequence_data,4);
    r = size(image_sequence.sequence_data,1);
    c = size(image_sequence.sequence_data,2);

    predicted_labels = zeros(r,c,num_img);
    selected_frames = [selected_frames num_img+1];
    for i=1:size(selected_frames,2)-1
        start_frame = selected_frames(i);
        end_frame   = selected_frames(i+1)-1;    
    
        %Since we are assuming that the start_frame is human annotated, simply
        %take the predicted_labels as sequence labels (oracle), so that the
        %error contributed by it will be zero
        predicted_labels(:,:,start_frame) = image_sequence.sequence_labels(:,:, start_frame);
    
        if(start_frame~=end_frame)                
            p_labels = label_propagation_oneway(image_sequence, start_frame, end_frame);
            predicted_labels(:,:,start_frame+1:end_frame) = p_labels;
        end        
    end
end

function [predicted_labels] = nearest_neighbor_propagation(selected_frames, image_sequence)
    
    num_img=size(image_sequence.sequence_data,4);
    r = size(image_sequence.sequence_data,1);
    c = size(image_sequence.sequence_data,2);
    
    predicted_labels = zeros(r,c,num_img); 
    
    for i=1:size(selected_frames,2)       
        sel_frame = selected_frames(i);
        predicted_labels(:,:,sel_frame) = image_sequence.sequence_labels(:,:, sel_frame);            
    end
    
    if(selected_frames(1)~=1)
        selected_frames = [-1 selected_frames];
    end
    if(selected_frames(end)~=num_img)
        selected_frames = [selected_frames -1];
    end
       
    for i=1:size(selected_frames,2)-1
        start_frame = selected_frames(i);
        end_frame = selected_frames(i+1);        
        if(start_frame == -1)
            p_labels = label_propagation_oneway(image_sequence, end_frame, 1);
            direction = -1;
            predicted_labels(:,:,end_frame+direction:direction:1) = p_labels;
        elseif(end_frame == -1)
            p_labels = label_propagation_oneway(image_sequence, start_frame, num_img);
            direction = 1;
            predicted_labels(:,:,start_frame+direction:direction:num_img) = p_labels;
        else            
            mid_frame = floor((start_frame + end_frame)/2);           
            p_labels = label_propagation_oneway(image_sequence, start_frame, mid_frame);
            direction = 1;
            predicted_labels(:,:,start_frame+direction:direction:mid_frame) = p_labels;
            p_labels = label_propagation_oneway(image_sequence, end_frame, mid_frame+1);
            direction = -1;
            predicted_labels(:,:,end_frame+direction:direction:mid_frame+1) = p_labels;
        end        
    end
end

function [predicted_labels] = pixelflow_propagation(selected_frames, image_sequence)
    
    num_img=size(image_sequence.sequence_data,4);
    r = size(image_sequence.sequence_data,1);
    c = size(image_sequence.sequence_data,2);
        
    predicted_labels = zeros(r,c,num_img);
    
    for i=1:size(selected_frames,2)       
        sel_frame = selected_frames(i);
        predicted_labels(:,:,sel_frame) = image_sequence.sequence_labels(:,:, sel_frame);            
    end
    
    if(selected_frames(1)~=1)
        selected_frames = [-1 selected_frames];
    end
    if(selected_frames(end)~=num_img)
        selected_frames = [selected_frames -1];
    end
       
    for i=1:size(selected_frames,2)-1
        lt = selected_frames(i);
        rt = selected_frames(i+1);            
        plabels = label_propagation_pixelflow(image_sequence, lt, rt);
        if(lt == -1)
            predicted_labels(:,:,1:rt-1) = plabels;
        elseif(rt == -1)
            predicted_labels(:,:,lt+1:end) = plabels;
        else
            predicted_labels(:,:,lt+1:rt-1) = plabels;
        end
    end        
end

function [predicted_labels] = mrf_propagation(selected_frames, image_sequence)
    
    num_img=size(image_sequence.sequence_data,4);
    r = size(image_sequence.sequence_data,1);
    c = size(image_sequence.sequence_data,2);
        
    image_sequence.appearance_model = ...
        train_appearance_model(selected_frames,image_sequence);
    
    predicted_labels = zeros(r,c,num_img);
    
    for i=1:size(selected_frames,2)       
        sel_frame = selected_frames(i);
        predicted_labels(:,:,sel_frame) = image_sequence.sequence_labels(:,:, sel_frame);            
    end
    
    if(selected_frames(1)~=1)
        selected_frames = [-1 selected_frames];
    end
    if(selected_frames(end)~=num_img)
        selected_frames = [selected_frames -1];
    end
       
    for i=1:size(selected_frames,2)-1
        lt = selected_frames(i);
        rt = selected_frames(i+1);                    
        plabels = mrf_label_propagation_pixelflow(image_sequence, lt, rt);
        if(lt == -1)
            predicted_labels(:,:,1:rt-1) = plabels;
        elseif(rt == -1)
            predicted_labels(:,:,lt+1:end) = plabels;
        else
            predicted_labels(:,:,lt+1:rt-1) = plabels;
        end
    end        
end

function [] = show_figures(image_sequence,predicted_labels)
sequence_labels = image_sequence.sequence_labels;
sequence_data = image_sequence.sequence_data;
figure;
for i=1:size(predicted_labels,3)    
    subplot(2,3,1), imagesc(sequence_labels(:,:,i)),title('Ground Truth');
    subplot(2,3,2), imagesc(predicted_labels(:,:,i)), title('Propagated Labels');
    diff_img = double(double(sequence_labels(:,:,i)) - predicted_labels(:,:,i));
    diff_img(find(diff_img~=0)) = 1;    
    subplot(2,3,3),imagesc(diff_img),title(['diff img:' num2str(i)]);
    subplot(2,3,4),imshow(sequence_data(:,:,:,i)), title('image');    
    pause(0.1);    
end
close all;
end


function [appearance_model] = train_appearance_model(selected_frames,image_sequence)
    sequence_labels = image_sequence.sequence_labels;
    sequence_data = image_sequence.sequence_data;
    unique_labels = unique(sequence_labels);
    num_classes = length(unique_labels);    
    num_img=size(image_sequence.sequence_data,4);
    
    
    appearance_model = cell(num_img,num_classes);
    
    options(14) = 100;
    options(1) = -1;    
    
    for sel_frame = selected_frames
        fprintf('Computing Appearance Model for Frame %d...\n',sel_frame);
        
        img = sequence_data(:,:,:,sel_frame); 
        entropy_img = entropyfilt(img);
        
        label_img = sequence_labels(:,:,sel_frame);
        unique_labels_sel_frame = unique(label_img);                
        
        for ind=unique_labels_sel_frame'                     
            class_pixels = find(label_img == ind );
            data=[];            
            for ch_ind=1:3                
                tmp = img(:,:,ch_ind);            
                data = [data double(tmp(class_pixels))];                
                tmp = entropy_img(:,:,ch_ind);
                data = [data double(tmp(class_pixels))];            
            end             
            if(isempty(data))
                data  = ones(1,6);
            end
            count = sel_frame;
            data = double(data);                 
            %fprintf('class %d...\n',ind);
            appearance_model{count,ind+1} = gmm(size(data,2), min(5, size(data,1)), 'full');
            %fprintf('initializaing gmm...\n');
            appearance_model{count,ind+1} = gmminit(appearance_model{count,ind+1}, data,options);
            %fprintf('running em...\n');
            appearance_model{count,ind+1} = gmmem(appearance_model{count,ind+1},data,options);
        end        
    end                        
end