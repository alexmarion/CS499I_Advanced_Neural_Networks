image_height = 5;
image_width = 5;

stride = 1;
window_size = 2;

[num_classes,classes,image_maps] = load_image_data(image_height,image_width);

[std_image_maps,m,s] = standardize_data(image_maps);

for w = 1:(image_width/window_size)
    start_pt = (w - 1) * stride + 1;
    end_pt = start_pt + window_size
    window = std_image_maps(:,:,1);
end