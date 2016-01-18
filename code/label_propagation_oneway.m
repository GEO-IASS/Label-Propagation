function [predicted_labels] = label_propagation_oneway(image_sequence, start_frame, end_frame)
%Method to propagate labels in either the forward direction or the
%backward direction.
%image_sequence contains the data about the image sequence, including
%actual images, forward flow, backward flow and groundtruth labels
%The implementation assumes that the startframe is labeled and it's labels
%have to be propagated upto and including endframe.
%startframe < endframe => labels to be propagated in the forward direction
%startframe > endframe => labels to be propagated in the backward direction

sequence_data = image_sequence.sequence_data;
sequence_labels = image_sequence.sequence_labels;
forward_flow = image_sequence.forward_flow;
backward_flow = image_sequence.backward_flow;
r = size(sequence_data,1);
c = size(sequence_data,2);
[orig_y,orig_x] = meshgrid(1:r, 1:c);
xx = orig_x(:); %Represents cols
yy = orig_y(:); %Represents rows

direction = sign(end_frame-start_frame);

predicted_labels = zeros(r,c,abs(end_frame - start_frame));

% Assuming that we are given the labels of start frame
prev_labels = sequence_labels(:,:,start_frame); 
    
count = 1;
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
    
    predicted_labels(:,:,count) = curr_labels;
    
    prev_labels = curr_labels;
    
    xx = orig_x(:); %Represents cols
    yy = orig_y(:); %Represents rows
    count = count+1;
end
end

