function [ num_classes,training_fields,training_classes,testing_fields,testing_classes ] = load_and_shuffle_data( image_size,data_selection_type )
    % Load the data
    [num_classes, classes, fields] = load_image_data(image_size,image_size);
    
    % Split into traiing and testing sets
    [training_fields,training_classes,testing_fields,testing_classes] = get_training_and_testing_sets(fields,classes,data_selection_type);
end