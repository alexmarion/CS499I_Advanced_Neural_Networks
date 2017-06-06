image_height = 100;
image_width = 100;

stride = 25;
window_size = 50;
num_horizontal_windows = length(1:stride:image_width - stride);
num_vertical_windows = length(1:stride:image_height - stride);

[num_classes,classes,image_maps] = load_image_data(image_height,image_width);

[std_image_maps,m,s] = standardize_data(image_maps);

% Slightly different for first iteration (because of indexing from 1)
height_start_pt = 1;
height_end_pt = height_start_pt + window_size - 1;
for h = 1:num_vertical_windows
    width_start_pt = 1;
    width_end_pt = width_start_pt + window_size - 1;
    fprintf('Height Start: %d\nHeight End: %d\n',height_start_pt,height_end_pt);
    for w = 1:num_horizontal_windows
        fprintf('Width Start: %d\nWidth End: %d\n',width_start_pt,width_end_pt);
        window = std_image_maps(height_start_pt:height_end_pt,width_start_pt:width_end_pt,1);
        
        %figure();
        %imshow(window);

        if w == 1
            width_start_pt = width_start_pt + stride - 1;
        else
            width_start_pt = width_start_pt + stride;
        end
        width_end_pt = width_start_pt + window_size;
    end
    if h == 1
        height_start_pt = height_start_pt + stride - 1;
    else
        height_start_pt = height_start_pt + stride;
    end
    height_end_pt = height_start_pt + window_size;
end

figure();
imshow(std_image_maps(:,:,1))