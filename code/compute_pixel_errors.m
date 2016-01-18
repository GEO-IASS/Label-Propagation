function [pixel_errors] = compute_pixel_errors(image_sequence, predicted_labels)
    sequence_labels = image_sequence.sequence_labels;    
    pixel_errors = [];
    for i=1:size(predicted_labels,3)       
        if(isfield(image_sequence,'superpixel_data'))
            sequence_labels(:,:,i) = smoothen(sequence_labels(:,:,i), ...
                        image_sequence.superpixel_data(:,:,i));
            predicted_labels(:,:,i) = smoothen(predicted_labels(:,:,i), ...
                        image_sequence.superpixel_data(:,:,i));        
        end
        diff_img = double(double(sequence_labels(:,:,i)) - predicted_labels(:,:,i));
        diff_img(find(diff_img~=0)) = 1;
        pixel_errors = [pixel_errors size(find(diff_img==1),1)];    
    end
end
