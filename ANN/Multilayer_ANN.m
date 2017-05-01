classdef Multilayer_ANN
    properties  
        num_classes
        num_hidden_node_layers
        num_hidden_nodes
        training_iters
        eta
        percent_field_retention
    end
    methods  
        function [ testing_accuracy,training_accuracy ] = train_ANN( ann,training_fields,training_classes,testing_fields,testing_classes )
            %% Control Flow Values
            should_add_bias_to_input = true;
            should_add_bias_to_hidden = false;
            should_std_data = true;
            should_perform_PCA = true;

            %% Set Initial Vals
            num_output_nodes = ann.num_classes;
            activation_fxn = @(x) 1./(1 + exp(-x));

            % Get number of training rows and number of testing rows
            num_training_rows = length(training_fields(:,1));
            num_testing_rows = length(testing_fields(:,1));

            num_data_cols = length(training_fields(1,:));

            %% Perform PCA
            if should_perform_PCA
                projection_vectors = PCA(training_fields,ann.percent_field_retention);
                training_fields = training_fields * projection_vectors;
                testing_fields = testing_fields * projection_vectors;

                num_data_cols = length(training_fields(1,:));
            end

            %% Standardize Data
            if should_std_data
                % Standardize data via training mean and training std dev
                [std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);

                std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
                std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
            else
                std_training_fields = training_fields;
                std_testing_fields = testing_fields;
            end

            %% Add bias nodes to input layer
            if should_add_bias_to_input
                % Add bias node and increase number of columns by 1
                std_training_fields = [ones(num_training_rows, 1), std_training_fields];
                std_testing_fields = [ones(num_testing_rows, 1), std_testing_fields];
                num_data_cols = num_data_cols + 1;
            end

            % Reformat training classes
            new_training_classes = zeros(num_training_rows,ann.num_classes);
            for i = 1:num_training_rows
                new_training_classes(i,training_classes(i)) = 1;
            end

            %% Perform Forward/Backward Propagation with Batch Gradient Descent
            iter = 0;

            % Initialize weights as random 
            range = [-1,1];
            beta = (range(2)-range(1)).*rand(num_data_cols, ann.num_hidden_nodes) + range(1);
            theta = (range(2)-range(1)).*rand(ann.num_hidden_nodes, num_output_nodes) + range(1);

            if should_add_bias_to_hidden
                % theta = [ones(num_hidden_nodes,1) theta];
                % beta = [ones(1,num_hidden_nodes);beta];
                beta = [ones(num_data_cols,1) beta];
                theta = [ones(1,num_output_nodes);theta];
            end

            % Matrix to track training error for plotting
            training_accuracy = zeros(ann.training_iters, 2);

            while iter < ann.training_iters
                iter = iter + 1;    
                %% Forward Propagation
                % Compute hidden layer
                training_h = activation_fxn(std_training_fields * beta);

                % Compute output layer    
                training_o = activation_fxn(training_h * theta);

                %% Backward Propagation
                % Compute output error
                delta_output = new_training_classes - training_o;

                % Update theta
                theta = theta + ((ann.eta/num_training_rows) * delta_output' * training_h)';

                % Compute hidden error
                delta_hidden = (theta * delta_output')' .* (training_h .* (1 - training_h));

                % Update beta
                beta = beta + (ann.eta/num_training_rows) * (delta_hidden' * std_training_fields)';

                % Choose maximum output node as value
                [~,training_o] = max(training_o,[],2);

                % Log training error
                num_correct = numel(find(~(training_classes - training_o)));
                acc = num_correct/num_training_rows;
                training_accuracy(iter,:) = [iter,acc];
            end

            %% Testing
            % Compute hidden layer
            testing_h = activation_fxn(std_testing_fields * beta);

            % Compute output layer
            testing_o = activation_fxn(testing_h * theta);

            % Choose maximum output node as value
            [~,testing_o] = max(testing_o,[],2);

            % Compute number of correct predictions
            num_correct = numel(find(~(testing_classes - testing_o)));
            testing_accuracy = num_correct/num_testing_rows;

            %{
            Plot the training error
            figure();
            plot(training_accuracy(:,1), training_accuracy(:,2));
            legend('Training Error');
            xlabel('Iteration');
            ylabel('Accuracy');
            %}
        end
    end
end