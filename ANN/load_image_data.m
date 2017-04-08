function [ num_classes, classes, fields ] = load_image_data()
    % image_map = double(rgb2gray(imread('./faces/00001/test.ppm')));
    % image_fields = reshape(image_map, [1,numel(image_map)]);

    root = './faces';
    class_folders = dir(root);
    class_folders = remove_dot_dirs(class_folders);

    data = [];

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
                image_map = double(rgb2gray(imread(fullfile(root,class_folder.name,img.name))));
                image_fields = reshape(image_map, [1,numel(image_map)]);
                % Append the class to the end of the fields
                image_fields(end + 1) = str2double(class_folder.name);

                data(end + 1,:) = image_fields;
            end
        end
    end
    
    num_classes = numel(class_folders);
    classes = data(:,end);
    fields = data(:,1:end-1);
end

