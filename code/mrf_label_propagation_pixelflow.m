function [predicted_labels] = mrf_label_propagation_pixelflow(image_sequence, lt, rt)

sequence_data = image_sequence.sequence_data;
sequence_labels = image_sequence.sequence_labels;
forward_flow = image_sequence.forward_flow;
backward_flow = image_sequence.backward_flow;
sequence_appearance_model = image_sequence.appearance_model;      

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
    
    
    fprintf('Propgating labels from frame %d to frame %d with GC \n',start_frame,end_frame);    
    for i=start_frame:direction:end_frame+(-1*direction)        
        from_frame = i;        
        to_frame = i+direction;
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
        
        curr_img = image_sequence.sequence_data(:,:,:,to_frame);                
        gt_appearance_model = sequence_appearance_model(start_frame,:);
        
        num_labels = length(gt_appearance_model);
        app_data_cost = get_data_cost(curr_img,gt_appearance_model,num_labels);
        
        flow_data_cost = zeros(r,c,num_labels);        
        
        label_cost = compute_flow_error_online(image_sequence,to_frame,...
                    from_frame,cost_options);
        
        
        flow_labels = zeros(r,c);
        flow_labels(idx) = prev_labels(idx_new);
                                
        uni_labels = unique(prev_labels);
        for lab_ind = uni_labels'
            label_idx = find(flow_labels == lab_ind);
            temp_cost = 10*ones(r,c);
            temp_cost(label_idx) = min(temp_cost(label_idx),label_cost(label_idx));
            flow_data_cost(:,:,lab_ind+1) = temp_cost;
        end
        
        data_cost = app_data_cost + flow_data_cost;
        smoothness_cost = ones(num_labels) - eye(num_labels);
        [Hc Vc] = colordiff(im2double(curr_img));
        [Hf Vf] = flowdiff(tracking_flow);
        
        %tic;
        %fprintf('calling graph cut open\n');
        %gch = GraphCut('open', data_cost, smoothness_cost,vC,hC);        
        gch = GraphCut('open', data_cost, smoothness_cost, (Vc+10*Vf), (Hc+10*Hf));
        %fprintf('calling graph cut expand\n');
        [gch curr_labels] = GraphCut('swap',gch);
        [gch se de] = GraphCut('energy', gch);        
        %fprintf('calling graph cut close\n');
        gch = GraphCut('close', gch);
        fprintf('.');
        %fprintf('done graph cutting\n');
        %toc;
        
        %curr_labels(idx) = prev_labels(idx_new);
        
        prev_labels = curr_labels;
        %xx = xx_new;
        %yy = yy_new;    
        xx = orig_x(:); %Represents cols
        yy = orig_y(:); %Represents rows        
        
        plabels(:,:,count) = curr_labels;        
        count = count+direction;                        
    end        
    fprintf('\n');
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


function [data_cost] = get_data_cost(img,apperance_model,num_labels)

[r,c, nch] = size(img);
probs = zeros(r, c, num_labels);
entropy_img = entropyfilt(img);

for i=1:length(apperance_model)
  data =[];
  for cl=1:nch
    tmp = img(:,:,cl);            
    data = [data double(tmp(:))];
    tmp = entropy_img(:,:,cl);
    data = [data double(tmp(:))];
  end  
  if(~isempty(apperance_model{i}))
    probs(:,:,i) = reshape(gmmprob(apperance_model{i}, data),r,c);
  end
end

probs(find(isnan(probs))) = 1e-20;
probs(find(probs == 0)) = 1e-20;

tot = sum(probs, 3);
tot(find(tot == 0)) = 1;
for i=1:length(apperance_model)
  probs(:,:,i) = probs(:,:,i)./tot;
end
probs(find(probs == 0)) = 1e-20;
[tmp, labels] = max(probs,[],3);
data_cost = -log(probs);
end

function [hC vC] = colordiff(im)
    vC = zeros(size(im,1),size(im,2));
    for c=1:3
      tmp = im(:,:,c);
      tmpim = [zeros(1,size(im,2)); tmp(1:end-1,:)];
      vC = vC + (tmpim - tmp).^2;
    end
    beta = 1/2/mean(vC(:)*5);
    vC = exp(-beta * vC);

    hC = zeros(size(im,1),size(im,2));
    for c=1:3
      tmp = im(:,:,c);
      tmpim = [zeros(size(im,1),1) tmp(:,1:end-1)];
      hC = hC + (tmpim - tmp).^2;
    end
    beta = 1/2/mean(hC(:)*5);
    hC = exp(-beta * hC);
end

function [hC vC] = flowdiff(im)

%tmp(:,:,1) = atand(im(:,:,2)./im(:,:,1));
%tmp(:,:,2) = sqrt(im(:,:,2).^2 + im(:,:,1).^2);
%im = tmp;

vC = zeros(size(im,1),size(im,2));
for c=1:2
  tmp = im(:,:,c);
  tmpim = [zeros(1,size(im,2)); tmp(1:end-1,:)];
  vC = vC + (tmpim - tmp).^2;
end
%vC = min(0.5, vC);
beta = 1/2/mean(vC(:));
vC = exp(-beta * vC);

hC = zeros(size(im,1),size(im,2));
for c=1:2
  tmp = im(:,:,c);
  tmpim = [zeros(size(im,1),1) tmp(:,1:end-1)];
  hC = hC + (tmpim - tmp).^2;
end
%hC = min(3, hC);
beta = 1/2/mean(hC(:));
hC = exp(-beta * hC);
end