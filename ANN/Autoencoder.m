classdef Autoencoder < ANN
    properties
        
    end
    methods
        function [ ann,training_accuracy,validation_accuracy,testing_accuracy ] = train_ANN( ann,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes )
            %% Control Flow Values
            % ann.should_add_bias_to_input = true;
            % ann.should_add_bias_to_hidden = false;
            ann.should_std_data = false;
            ann.should_perform_PCA = false;

            %% Set Initial Vals
            num_output_nodes = length(training_fields);

            % Get number of training rows and number of testing rows
            num_training_rows = length(training_fields(:,1));
            num_validation_rows = length(validation_fields(:,1));
            num_testing_rows = length(testing_fields(:,1));

            num_data_cols = length(training_fields(1,:));

            %% Perform PCA
            if ann.should_perform_PCA
                projection_vectors = PCA(training_fields,ann.percent_field_retention);
                training_fields = training_fields * projection_vectors;
                validation_fields = validation_fields * projection_vectors;
                testing_fields = testing_fields * projection_vectors;

                num_data_cols = length(training_fields(1,:));
                num_output_nodes = size(training_fields,2);

            end
            
            %% Standardize Data
            if ann.should_std_data
                % Standardize data via training mean and training std dev
                [std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);

                std_validation_fields = validation_fields - repmat(training_fields_mean,size(validation_fields,1),1);
                std_validation_fields = std_validation_fields ./ repmat(training_fields_std_dev,size(std_validation_fields,1),1);
                
                std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
                std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
            else
                std_training_fields = training_fields;
                std_validation_fields = validation_fields;
                std_testing_fields = testing_fields;
            end
            
            %% Add noise to input layer
            std_output_fields = std_training_fields;
            std_training_fields = std_training_fields + randn(size(std_training_fields));
            
            

            %% Add bias nodes to input layer
            
            % Add bias node and increase number of columns by 1
            std_training_fields = [ones(num_training_rows, 1), std_training_fields];
            std_validation_fields = [ones(num_validation_rows, 1), std_validation_fields];
            std_testing_fields = [ones(num_testing_rows, 1), std_testing_fields];
            num_data_cols = num_data_cols + 1;
           

            %% Perform Forward/Backward Propagation with Batch Gradient Descent
            iter = 0;

            % Initialize weights as random 
            range = [-1,1];
            ann.beta = (range(2)-range(1)).*rand(num_data_cols, ann.num_hidden_nodes) + range(1);
            ann.theta = (range(2)-range(1)).*rand(ann.num_hidden_nodes, num_output_nodes) + range(1);

            %% Add bias nodes to hidden layer
            ann.beta = [ones(num_data_cols,1) ann.beta];
            ann.theta = [ones(1,num_output_nodes);ann.theta];

            % Matrix to track training error for plotting
            training_accuracy = zeros(ann.training_iters, 2);

            while iter < 10000
                iter = iter + 1;    
                %% Forward Propagation
                % Compute hidden layer
                training_h = ann.activation_fxn(std_training_fields * ann.beta);

                % Compute output layer    
                training_o = ann.activation_fxn(training_h * ann.theta);

                %% Backward Propagation
                % Compute output error
                delta_output =  std_training_fields(:,1:end-1) - training_o;

                % Update theta
                ann.theta = ann.theta + ((ann.eta/num_training_rows) * delta_output' * training_h)';

                % Compute hidden error
                delta_hidden = (ann.theta * delta_output')' .* (training_h .* (1 - training_h));

                % Update beta
                ann.beta = ann.beta + (ann.eta/num_training_rows) * (delta_hidden' * std_training_fields)';

                % Choose maximum output node as value
                [~,training_o] = max(training_o,[],2);

                % Log training error
                num_correct = numel(find(~(std_training_fields(:,1:end-1) - training_o)));
                acc = num_correct/num_training_rows;
                training_accuracy(iter,:) = [iter,acc];
                
                % Decide learning rate dynamically                
%                 if iter >= 3  && (acc - training_accuracy(iter - 1,2)) < (training_accuracy(iter - 1,2) - training_accuracy(iter - 2,2))
%                     ann.eta = ann.eta/2;
%                 else
%                     ann.eta = ann.eta + 0.05;
%                 end
            end
            
            ann.theta = mean(ann.theta, 2); 
            new_training_classes = zeros(num_training_rows,ann.num_classes);
            
            
            %% Validation
            validation_accuracy = test_ANN(ann,std_validation_fields,validation_classes);

            %% Testing
            testing_accuracy = test_ANN(ann,std_testing_fields,testing_classes);
            
            if ann.should_plot_train
                % Plot the training error
                figure();
                plot(training_accuracy(:,1), training_accuracy(:,2));
                legend('Training Accuracy');
                xlabel('Iteration');
                ylabel('Accuracy');
                fprintf('Validation Accuracy: %f%%\n',validation_accuracy * 100);
                fprintf('Testing Accuracy: %f%%\n',testing_accuracy * 100);
            end  
        end
    end
end