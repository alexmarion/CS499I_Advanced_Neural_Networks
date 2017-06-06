image_height = 100;
image_width = 100;

stride = 25;
window_size = 50;
num_horizontal_windows = length(1:stride:image_width - stride);
num_vertical_windows = length(1:stride:image_height - stride);

[num_classes,classes,image_maps] = load_image_data(image_height,image_width);

[std_image_maps,m,s] = standardize_data(image_maps);

% Slightly different for first iteration (because of indexing from 1)
start_pt = 1;
end_pt = start_pt + window_size - 1;
for w = 1:num_vertical_windows
    fprintf('Start: %d\nEnd: %d\n',start_pt,end_pt);
    window = std_image_maps(start_pt:end_pt,1:window_size,1);
    figure();
    imshow(window);
    
    if w == 1
        start_pt = start_pt + stride - 1;
    else
        start_pt = start_pt + stride;
    end
    end_pt = start_pt + window_size;
end

figure();
imshow(std_image_maps(:,:,1))