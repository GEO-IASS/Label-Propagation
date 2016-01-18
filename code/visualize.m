%% Visualize

h1 = implay(image_sequence.sequence_data); set(h1.Parent, 'Name', 'Original Video') %// set title
h2 = implay(image_sequence.sequence_labels); title('Actual Labels'); set(h2.Parent, 'Name', 'Actual Label') %// set title
h3 = implay(predicted_labels); title('Predicted Labels'); set(h3.Parent, 'Name', 'Predicted Label') %// set title

% nSeq = size(image_sequence.sequence_data,4);
% 
% for i = 1 : nSeq
%     im = squeeze(image_sequence.sequence_data(:,:,:,i));
%     figure(i);imshow(im);
% end
    