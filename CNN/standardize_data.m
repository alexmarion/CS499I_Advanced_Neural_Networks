function [ std_image_maps, m, s ] = standardize_data( image_maps )
%standardize_data standardizes image maps passed in an X*Y*Z matrix
    % Take the mean of each feature in the image
    m = mean(image_maps, 3);
    
    s = std(image_maps,[],3);
    
    % Avoid dividing by 0
    s(s == 0) = 1;
    
    std_image_maps = image_maps - m;
    std_image_maps = std_image_maps ./ s;
end

