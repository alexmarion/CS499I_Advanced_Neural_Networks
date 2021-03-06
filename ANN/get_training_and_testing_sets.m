function [ training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes ] = get_training_and_testing_sets( fields,classes )
    %% Default: 2/3 Training 1/3 Testing
    % 80% of training data goes to training, 20% to validation
    % Initialize vars
    num_data_rows = length(fields(:,1));

    % Concatinate data for shuffling and randomly permutate
    fields_and_classes = [fields classes];
    shuffled_fields_and_classes = fields_and_classes(randperm(num_data_rows),:);
    shuffled_fields = shuffled_fields_and_classes(:,1:end-1);
    shuffled_classes = shuffled_fields_and_classes(:,end);

    % Find index of top 2/3
    two_thirds = ceil((num_data_rows/3)*2);

    % Extract top 2/3 for training, buttom 1/3 for testing
    training_fields = shuffled_fields(1:two_thirds,:);
    training_classes = shuffled_classes(1:two_thirds,:);
    
    testing_fields = shuffled_fields(two_thirds+1:end,:);
    testing_classes = shuffled_classes(two_thirds+1:end,:);
    
    % Find index of top 80%
    training_percent = ceil(length(training_classes) * 0.80);
    
    validation_fields = training_fields(training_percent+1:end,:);
    validation_classes = training_classes(training_percent+1:end,:);
    
    training_fields = training_fields(1:training_percent,:);
    training_classes = training_classes(1:training_percent,:);
end

