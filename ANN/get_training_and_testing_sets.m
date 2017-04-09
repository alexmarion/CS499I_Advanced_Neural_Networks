function [ training_fields,training_classes,testing_fields,testing_classes ] = get_training_and_testing_sets( fields,classes,type )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    % Optional argument 'type' is set to default if not supplied
    if ~exist('type','var') || isempty(type)
      type = 0;
    end
    
    if type == 1
        %% Type 1: Leave One Out
        
    else
        %% Default: 2/3 Training 1/3 Testing
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
    end
end

