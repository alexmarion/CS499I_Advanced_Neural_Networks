function [ training_maps,training_classes,validation_maps,validation_classes,testing_maps,testing_classes ] = get_training_and_testing_sets( image_maps,classes )

    %% Default: 2/3 Training 1/3 Testing
    % 80% of training data goes to training, 20% to validation
    % Initialize vars
    num_data_rows = size(image_maps,3);

    % Concatinate data for shuffling and randomly permutate
    r = randperm(num_data_rows);
    shuffled_image_maps = image_maps(:,:,r);
    shuffled_classes = classes(r);

    % Find index of top 2/3
    two_thirds = ceil((num_data_rows/3)*2);

    % Extract top 2/3 for training, buttom 1/3 for testing
    training_maps = shuffled_image_maps(:,:,1:two_thirds);
    training_classes = shuffled_classes(1:two_thirds,:);
    
    testing_maps = shuffled_image_maps(:,:,two_thirds+1:end);
    testing_classes = shuffled_classes(two_thirds+1:end,:);
    
    % Find index of top 80%
    training_percent = ceil(length(training_classes) * 0.80);
    
    validation_maps = training_maps(:,:,training_percent+1:end);
    validation_classes = training_classes(training_percent+1:end,:);
    
    training_maps = training_maps(:,:,1:training_percent);
    training_classes = training_classes(1:training_percent,:);
end

