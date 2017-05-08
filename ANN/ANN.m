classdef ANN
    properties
        % Control Flow Values
        should_add_bias_to_input = true
        should_add_bias_to_hidden = false
        should_std_data = true
        should_perform_PCA = true
        
        % Parameters
        num_classes = 15
        num_hidden_nodes = 20
        training_iters = 1000
        eta = 0.5
        percent_field_retention = 0.95
        
        % Values
        beta
        theta
        activation_fxn = @(x) 1./(1 + exp(-x))
    end
    methods
        function ann = set_control_flow_vals(ann, vals)
            ann.should_add_bias_to_input = vals(1);
            ann.should_add_bias_to_hidden = vals(2);
            ann.should_std_data = vals(3);
            ann.should_perform_PCA = vals(4);
        end
        function [ testing_accuracy,training_accuracy,ann ] = train_ANN( ann,training_fields,training_classes,testing_fields,testing_classes )
            %% Control Flow Values
            % ann.should_add_bias_to_input = true;
            % ann.should_add_bias_to_hidden = false;
            % ann.should_std_data = true;
            % ann.should_perform_PCA = true;

            %% Set Initial Vals
            num_output_nodes = ann.num_classes;

            % Get number of training rows and number of testing rows
            num_training_rows = length(training_fields(:,1));
            num_testing_rows = length(testing_fields(:,1));

            num_data_cols = length(training_fields(1,:));

            %% Perform PCA
            if ann.should_perform_PCA
                projection_vectors = PCA(training_fields,ann.percent_field_retention);
                training_fields = training_fields * projection_vectors;
                testing_fields = testing_fields * projection_vectors;

                num_data_cols = length(training_fields(1,:));
            end

            %% Standardize Data
            if ann.should_std_data
                % Standardize data via training mean and training std dev
                [std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);

                std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
                std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
            else
                std_training_fields = training_fields;
                std_testing_fields = testing_fields;
            end

            %% Add bias nodes to input layer
            if ann.should_add_bias_to_input
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
            ann.beta = (range(2)-range(1)).*rand(num_data_cols, ann.num_hidden_nodes) + range(1);
            ann.theta = (range(2)-range(1)).*rand(ann.num_hidden_nodes, num_output_nodes) + range(1);

            if ann.should_add_bias_to_hidden
                % theta = [ones(num_hidden_nodes,1) theta];
                % beta = [ones(1,num_hidden_nodes);beta];
                ann.beta = [ones(num_data_cols,1) ann.beta];
                ann.theta = [ones(1,num_output_nodes);ann.theta];
            end

            % Matrix to track training error for plotting
            training_accuracy = zeros(ann.training_iters, 2);

            while iter < ann.training_iters
                iter = iter + 1;    
                %% Forward Propagation
                % Compute hidden layer
                training_h = ann.activation_fxn(std_training_fields * ann.beta);

                % Compute output layer    
                training_o = ann.activation_fxn(training_h * ann.theta);

                %% Backward Propagation
                % Compute output error
                delta_output = new_training_classes - training_o;

                % Update theta
                ann.theta = ann.theta + ((ann.eta/num_training_rows) * delta_output' * training_h)';

                % Compute hidden error
                delta_hidden = (ann.theta * delta_output')' .* (training_h .* (1 - training_h));

                % Update beta
                ann.beta = ann.beta + (ann.eta/num_training_rows) * (delta_hidden' * std_training_fields)';

                % Choose maximum output node as value
                [~,training_o] = max(training_o,[],2);

                % Log training error
                num_correct = numel(find(~(training_classes - training_o)));
                acc = num_correct/num_training_rows;
                training_accuracy(iter,:) = [iter,acc];
            end

            %% Testing
            % Compute hidden layer
            testing_h = ann.activation_fxn(std_testing_fields * ann.beta);

            % Compute output layer
            testing_o = ann.activation_fxn(testing_h * ann.theta);

            % Choose maximum output node as value
            [~,testing_o] = max(testing_o,[],2);

            % Compute number of correct predictions
            num_correct = numel(find(~(testing_classes - testing_o)));
            testing_accuracy = num_correct/num_testing_rows;

            %{
            % Plot the training error
            figure();
            plot(training_accuracy(:,1), training_accuracy(:,2));
            legend('Training Accuracy');
            xlabel('Iteration');
            ylabel('Accuracy');
            %}
        end
        function [ testing_accuracy ] = test_ANN( ann,testing_fields,testing_classes )
            %% Testing
            % Compute hidden layer
            testing_h = ann.activation_fxn(testing_fields * ann.beta);

            % Compute output layer
            testing_o = ann.activation_fxn(testing_h * ann.theta);

            % Choose maximum output node as value
            [~,testing_o] = max(testing_o,[],2);

            % Compute number of correct predictions
            num_correct = numel(find(~(testing_classes - testing_o)));
            testing_accuracy = num_correct/num_testing_rows;
        end
        function [] = plot_training_accuracy( ann,testing_accuracy,training_accuracy )
            fig = figure();
            plot(training_accuracy(:,1), training_accuracy(:,2));
            legend('Training Accuracy');
            xlabel('Iteration');
            ylabel('Accuracy');
            c_flow_str = control_flow_str(ann);
            title(c_flow_str);
            saveas(fig,sprintf('../Latex/accuracy_imgs/%s_training_accuracy.png',c_flow_str));
            
            % The following print is for the latex file
            fprintf('\\testingAccuracyTableAndPlot{%s}{%s}{%s}{%s}{%f}\n',...
                 ANN.binary_to_str(ann.should_add_bias_to_input),...
                 ANN.binary_to_str(ann.should_add_bias_to_hidden),...
                 ANN.binary_to_str(ann.should_std_data),...
                 ANN.binary_to_str(ann.should_perform_PCA),...
                 testing_accuracy);
        end
        function [ s_training_accuracies,s_testing_accuracies,ann ] = cross_validate_ANN( ann,S,fields,classes )
            num_data_rows = size(fields,1);
            s_folds = cvpartition(num_data_rows,'k',S);

            shuffled_idxs = randperm(num_data_rows);
            shuffled_classes = classes(shuffled_idxs);
            shuffled_fields = fields(shuffled_idxs,:);

            s_training_accuracies = zeros(S,2);
            s_testing_accuracies = zeros(S,2);

            for i=1:S
                idxs = training(s_folds,i);

                training_idxs = find(idxs);
                s_training_fields = shuffled_fields(training_idxs,:);
                s_training_classes = shuffled_classes(training_idxs);

                testing_idxs = find(~idxs);
                s_testing_fields = shuffled_fields(testing_idxs,:);
                s_testing_classes = shuffled_classes(testing_idxs);
                
                [testing_accuracy,training_accuracy,ann] = train_ANN(ann,s_training_fields,s_training_classes,s_testing_fields,s_testing_classes);

                s_training_accuracies(i,:) = [i,training_accuracy(end,2)];
                s_testing_accuracies(i,:) = [i,testing_accuracy];
            end
            %{
            figure();
            hold on;
            % plot(s_training_accuracies(:,1), s_training_accuracies(:,2),'b');
            % plot(s_testing_accuracies(:,1), s_testing_accuracies(:,2),'r');
            s_training_and_testing_accuracies = [s_training_accuracies(:,2) s_testing_accuracies(:,2)];
            bar(s_training_and_testing_accuracies);
            %bar(s_testing_accuracies(:,2),'r');
            legend('Training Accuracy','Testing Accuracy','Location','southwest')
            xlabel('Fold');
            ylabel('Accuracy');
            hold off;
            %}
        end
        function [ s_training_accuracies,s_testing_accuracies ] = dual_validate_ANN( ann,S,fields,classes )
            [ folding_fields, ...
              folding_classes, ...
              validation_fields, ...
              validation_classes ] = get_training_and_testing_sets(fields,classes);
            
            num_data_rows = size(folding_fields,1);
            s_folds = cvpartition(num_data_rows,'k',S);

            shuffled_idxs = randperm(num_data_rows);
            shuffled_classes = folding_classes(shuffled_idxs);
            shuffled_fields = folding_fields(shuffled_idxs,:);

            s_training_accuracies = zeros(S,2);
            s_testing_accuracies = zeros(S,2);

            for i=1:S
                idxs = training(s_folds,i);

                training_idxs = find(idxs);
                s_training_fields = shuffled_fields(training_idxs,:);
                s_training_classes = shuffled_classes(training_idxs);

                testing_idxs = find(~idxs);
                s_testing_fields = shuffled_fields(testing_idxs,:);
                s_testing_classes = shuffled_classes(testing_idxs);
                
                % Compute initial ANN
                [~,~,ann] = train_ANN(ann,s_training_fields,s_training_classes,s_testing_fields,s_testing_classes);
                
                % Validation set parameters
                val_learning_rates = 0.25:0.25:10;
                val_hidden_nodes = 20:1:400;

                val_accuracies = zeros(numel(val_learning_rates) * numel(val_hidden_nodes) , 3);
                
                % Use validation set to fine tune learning rate (eta) and
                % number of hidden nodes
                idx = 1;
                for lr=1:numel(val_learning_rates)
                    ann.eta = val_learning_rates(lr);
                    for hn=1:numel(val_hidden_nodes)
                        ann.num_hidden_nodes = val_hidden_nodes(hn);
                        [validation_accuracy,~,ann] = train_ANN(ann,s_training_fields,s_training_classes,validation_fields,validation_classes);
                        val_accuracies(idx,:) = [ann.eta, ann.num_hidden_nodes, validation_accuracy];
                        idx = idx + 1;
                    end
                end
                
                % Get optimal values
                [~,optimal_idx] = max(val_accuracies(:,3));
                ann.eta = val_accuracies(optimal_idx,1);
                ann.num_hidden_nodes = val_learning_rates(optimal_idx,2);
                
                % Retrain with optimal values
                [testing_accuracy,training_accuracy,ann] = train_ANN(ann,s_training_fields,s_training_classes,s_testing_fields,s_testing_classes);
                s_training_accuracies(i,:) = [i,training_accuracy(end,2)];
                s_testing_accuracies(i,:) = [i,testing_accuracy];
            end
        end
        
        function s = control_flow_str( ann )
            s1 = ANN.binary_to_str(ann.should_add_bias_to_input);
            s2 = ANN.binary_to_str(ann.should_add_bias_to_hidden);
            s3 = ANN.binary_to_str(ann.should_std_data);
            s4 = ANN.binary_to_str(ann.should_perform_PCA);
            s = strcat(s1,s2,s3,s4);
        end
    end
    methods(Static)        
        function s = binary_to_str( val )
            if val == 0
                s = 'N';
            else
                s = 'Y';
            end
        end
    end
end