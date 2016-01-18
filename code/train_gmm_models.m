function [appearance_model] = train_gmm_models(image_sequence)
    sequence_labels = image_sequence.sequence_labels;
    sequence_data = image_sequence.sequence_data;
    
    num_img=size(sequence_data,4);
    
    unique_labels = unique(sequence_labels);
    num_classes = length(unique_labels);    
    appearance_model = cell(num_img,num_classes);
    
    options(14) = 100;
    options(1) = -1;    

    disp(num_img);
    for sel_frame = 1:num_img        
        disp(sel_frame);
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
            data = double(data);                 
            %fprintf('class %d...\n',ind);
            appearance_model{sel_frame,ind+1} = gmm(size(data,2), min(5, size(data,1)), 'full');
            %fprintf('initializaing gmm...\n');
            appearance_model{sel_frame,ind+1} = gmminit(appearance_model{sel_frame,ind+1}, data,options);
            %fprintf('running em...\n');
            appearance_model{sel_frame,ind+1} = gmmem(appearance_model{sel_frame,ind+1},data,options);
        end        
    end                        
end