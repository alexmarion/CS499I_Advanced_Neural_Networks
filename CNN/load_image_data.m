function [ num_classes, classes, image_maps ] = load_image_data(image_height,image_width)
    % image_map = double(rgb2gray(imread('./faces/00001/test.ppm')));
    % image_fields = reshape(image_map, [1,numel(image_map)]);

    root = './yalefaces';
    class_folders = dir(root);
    class_folders = remove_dot_dirs(class_folders);

    image_maps = [];
    classes = [];

    % Loop through all class folders
    for class_folder = class_folders'
        current_dir = dir(fullfile(root,class_folder.name));
        % disp(class_folder.name)
        % Get all images from current class folder
        for imgs = current_dir
            current_imgs = remove_dot_dirs(imgs);
            % Loop through each image in current class folder
            for img = current_imgs'
                % Convert image to grayscale and load each pixel value into a field
                % image_map = double(rgb2gray(imread(fullfile(root,class_folder.name,img.name))));
                % image_map = double(imread(fullfile(root,class_folder.name,img.name)));
                
                image = imread(fullfile(root,class_folder.name,img.name));
                image_map = double(imresize(image,[image_height image_width]));
                image_maps = cat(3,image_maps,image_map);

                classes(end+1) = str2double(class_folder.name);
            end
        end
    end
    
    num_classes = numel(class_folders);
end

