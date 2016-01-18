function [predicted_labels] = label_propagation_pixelflow(image_sequence, lt, rt)

sequence_data = image_sequence.sequence_data;
sequence_labels = image_sequence.sequence_labels;
forward_flow = image_sequence.forward_flow;
backward_flow = image_sequence.backward_flow;
cost_options = image_sequence.cost_options;

r = size(sequence_data,1);
c = size(sequence_data,2);
num_img = size(sequence_data,4);

[orig_y,orig_x] = meshgrid(1:r, 1:c);

if(lt == -1)
    num_frames = rt - 1;
    terminal_frames = [-1 1];
elseif(rt == -1)
    num_frames = num_img - lt;
    terminal_frames = [num_img -1];
else
    num_frames = rt-lt-1;
    terminal_frames = [rt-1 lt+1];
end

predicted_labels = zeros(r,c,num_frames);

labels = cell(1,2);
labels{1} = zeros(r,c,num_frames);
labels{2} = zeros(r,c,num_frames);

labeled_frames = [lt rt];

for k=1:size(labels,2)
    xx = orig_x(:); %Represents cols
    yy = orig_y(:); %Represents rows
    
    if(labeled_frames(k)==-1)
        continue;
    end    
    start_frame = labeled_frames(k);
    end_frame = terminal_frames(k);        
    direction = sign(end_frame-start_frame);
    %disp([start_frame end_frame direction]);
    
    plabels = zeros(r,c,num_frames);

    % Assuming that we are given the labels of start frame
    prev_labels = sequence_labels(:,:,start_frame); 
    
    if(direction == 1) %Propagating labels forward
        count = 1;
    else
        count = num_frames;
    end
    
    
    %fprintf('Propgating labels from frame %d to frame %d\n',start_frame,end_frame);    
    for i=start_frame:direction:end_frame+(-1*direction)
        
        from_frame = i;        
        if(direction == 1) %Propagating labels forward
            tracking_flow = backward_flow(:,:,:,from_frame); 
        else
            tracking_flow = forward_flow(:,:,:,from_frame-1);
        end

        idx = sub2ind([r c], yy, xx);    
        tracking_flow_u = tracking_flow(:,:,1);
        tracking_flow_v = tracking_flow(:,:,2);
        
        xx_new = round(xx + tracking_flow_u(idx));
        yy_new = round(yy + tracking_flow_v(idx));
                    
        xx_new(find(xx_new <= 0)) = 1;
        yy_new(find(yy_new <= 0)) = 1;
        xx_new(find(xx_new > c)) = c;
        yy_new(find(yy_new > r)) = r;  

        idx_new = sub2ind([r c], yy_new, xx_new);  
    
        curr_labels = zeros(r,c);
        curr_labels(idx) = prev_labels(idx_new);
        
        prev_labels = curr_labels;
        
        xx = orig_x(:); %Represents cols
        yy = orig_y(:); %Represents rows        
        
        plabels(:,:,count) = curr_labels;        
        count = count+direction;                        
    end        
    labels{k} = plabels;
end
if(lt == -1)
    predicted_labels = labels{2};
elseif(rt == -1)
    predicted_labels = labels{1};
else        
    l_labels = labels{1};
    r_labels = labels{2};    
    count = 1;    
    for i=lt+1:rt-1                
        l_lab = l_labels(:,:,count);
        r_lab = r_labels(:,:,count);                
                
        l_cost = compute_flow_error_online(image_sequence,i,lt,cost_options);        
        r_cost = compute_flow_error_online(image_sequence,i,rt,cost_options);
                                
        label_map = zeros(r,c);
        label_map(find(l_cost-r_cost<=0)) = 1;
        label_map(find(l_cost-r_cost>0)) =2;    
        
        temp_labels = zeros(r,c);
        temp_labels(find(label_map == 1)) = l_lab(find(label_map == 1));
        temp_labels(find(label_map == 2)) = r_lab(find(label_map == 2));
        
        predicted_labels(:,:,count) = temp_labels;
        count = count + 1;
    end
end

end

