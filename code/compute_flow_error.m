function [cumulative_cost] = compute_flow_error(image_sequence,frame_number,options)

sequence_data = image_sequence.sequence_data;
forward_flow = image_sequence.forward_flow;
backward_flow = image_sequence.backward_flow;
num_img=size(sequence_data,4);

show_results = options.show_cost_plot;

% If the required frame is invalid
if(frame_number>num_img)
    disp('Frame number is invalid');
    return;
end

r = size(sequence_data,1);
c = size(sequence_data,2);
[orig_y,orig_x] = meshgrid(1:r, 1:c);

%forward_flow(:,:,:,i) represents the flow from i to i+1
%backward_flow(:,:,:,i) represents the flow from i+1 to i
frame_index = cell(2,1);
frame_index{1} = frame_number+1:num_img;
frame_index{2} = frame_number-1:-1:1;

cumulative_cost = zeros(1,num_img);
for i=1:size(frame_index,1)
    prev_prob = zeros(r,c);
    to_frame = frame_number;  
    xx = orig_x(:); %Represents cols
    yy = orig_y(:); %Represents rows
    for j=1:size(frame_index{i},2)
        from_frame = frame_index{i}(j);        
        
        if(i==1)
            tracking_flow = forward_flow(:,:,:,to_frame);  %Need this to track pixels
            reverse_flow  = backward_flow(:,:,:,to_frame); %Need this to detect occlusions
            if(from_frame == num_img)
                next_flow = tracking_flow;
            else
                next_flow = forward_flow(:,:,:,from_frame);    %Need this to find motion errors
            end
        else
            tracking_flow = backward_flow(:,:,:,to_frame-1);  %Need this to track pixels
            reverse_flow  = forward_flow(:,:,:,to_frame-1); %Need this to detect occlusions
            if(from_frame == 1)
                next_flow = tracking_flow;
            else
                next_flow = backward_flow(:,:,:,from_frame-1);    %Need this to find motion errors
            end
        end
                        
        idx = sub2ind([r c], orig_y(:), orig_x(:));    
        tracking_flow_u = tracking_flow(:,:,1);
        tracking_flow_v = tracking_flow(:,:,2);
        
        xx_new = round(xx + tracking_flow_u(idx));
        yy_new = round(yy + tracking_flow_v(idx));
        
        curr_cost = zeros(r,c);
                
        xx_new(find(xx_new <= 0)) = 1;
        yy_new(find(yy_new <= 0)) = 1;
        xx_new(find(xx_new > c)) = c;
        yy_new(find(yy_new > r)) = r;  
        
        idx_new = sub2ind([r c], yy_new, xx_new);    
        
        if(options.use_color == 1)                        
            %Penalizing for change in appearance        
            for img_dim=1:3
                to_img = double(sequence_data(:,:,img_dim,to_frame));
                from_img = double(sequence_data(:,:,img_dim,from_frame));
                curr_cost(idx) = curr_cost(idx) + (to_img(idx)-from_img(idx_new)).^2/256/256/3;                
            end       
        end
        
        if(options.use_motion == 1)
            for flow_dim=1:2
                to_flow = double(tracking_flow(:,:,flow_dim));
                from_flow = double(next_flow(:,:,flow_dim));            
                val = 1+to_flow(idx).^2+from_flow(idx_new).^2;                
                curr_cost(idx) = curr_cost(idx) + (to_flow(idx)-from_flow(idx_new)).^2./val*5;        
            end
        end
                
        
        %Penalizing for occlusions
        if(options.use_discrepancy == 1)
            discrepancy = tracking_flow + reverse_flow;
            discrepancy = discrepancy(:,:,1).^2 + discrepancy(:,:,2).^2;
            discrepancy = discrepancy ./ (tracking_flow(:,:,1).^2 + ...
                tracking_flow(:,:,2).^2+reverse_flow(:,:,1).^2 + reverse_flow(:,:,2).^2+1); 
            curr_cost = curr_cost + discrepancy;    
        end
                
        curr_prob = prev_prob + ((1-prev_prob).*(1-exp(-curr_cost)));                
        cumulative_cost(from_frame) = sum(sum(curr_prob));                
        if(show_results ~= 0)
            imagesc(curr_prob);
            pause(0.1);                                               
        end
        
        xx = xx_new;
        yy = yy_new;
        prev_prob = curr_prob;
        to_frame = from_frame;
    end
end

close all;
end
