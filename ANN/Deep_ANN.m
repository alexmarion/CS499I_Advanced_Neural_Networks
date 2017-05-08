classdef Deep_ANN
    properties
        % Parameters
        num_classes = 15;
        hidden_layer_division = 3;
        training_iters = 1000;
        eta = 0.5;
        percent_field_retention = 0.95;

        % Control Flow Values
        should_add_bias_to_input = false;
        should_add_bias_to_hidden = true;
        should_std_data = true;
        should_perform_PCA = false;
    end
    methods
        function deep_ann = set_control_flow_vals(deep_ann, vals)
            deep_ann.should_add_bias_to_input = vals(1);
            deep_ann.should_add_bias_to_hidden = vals(2);
            deep_ann.should_std_data = vals(3);
            deep_ann.should_perform_PCA = vals(4);
        end
        function [ testing_accuracy,training_accuracy ] = train_deep_ANN( deep_ann,training_fields,training_classes,testing_fields,testing_classes )
            %% Set Initial Vals
            num_output_nodes = deep_ann.num_classes;
            activation_fxn = @(x) 1./(1 + exp(-x));

            % Get number of training rows and number of testing rows
            num_training_rows = length(training_fields(:,1));
            num_testing_rows = length(testing_fields(:,1));

            num_data_cols = length(training_fields(1,:));

            %% Compute number of hidden layers based on input size and division ratio
            num_hidden_node_layers = 0;
            num_nodes_in_layer = num_data_cols;
            for i = 1:num_testing_rows
                num_hidden_node_layers = num_hidden_node_layers + 1;
                num_nodes_in_layer = ceil(num_nodes_in_layer/deep_ann.hidden_layer_division);
                if num_nodes_in_layer <= num_output_nodes
                    break
                end
            end

            %% Perform PCA
            if deep_ann.should_perform_PCA
                projection_vectors = PCA(training_fields,deep_ann.percent_field_retention);
                training_fields = training_fields * projection_vectors;
                testing_fields = testing_fields * projection_vectors;

                num_data_cols = length(training_fields(1,:));
            end

            %% Standardize Data
            if deep_ann.should_std_data
                % Standardize data via training mean and training std dev
                [std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);

                std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
                std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
            else
                std_training_fields = training_fields;
                std_testing_fields = testing_fields;
            end

            %% Add bias nodes to input layer
            if deep_ann.should_add_bias_to_input
                % Add bias node and increase number of columns by 1
                std_training_fields = [ones(num_training_rows, 1), std_training_fields];
                std_testing_fields = [ones(num_testing_rows, 1), std_testing_fields];
                num_data_cols = num_data_cols + 1;
            end

            % Reformat training classes
            new_training_classes = zeros(num_training_rows,deep_ann.num_classes);
            for i = 1:num_training_rows
                new_training_classes(i,training_classes(i)) = 1;
            end

            %% Perform Forward/Backward Propagation with Batch Gradient Descent
            % Initialize weights as random    
            range = [-1,1];
            weights = cell(num_hidden_node_layers,1);
            new_layer_size = ceil(num_data_cols/deep_ann.hidden_layer_division);
            weights{1} = (range(2)-range(1)).*rand(num_data_cols, new_layer_size)+range(1);

            for layer = 2:num_hidden_node_layers - 1 
                new_layer_size = ceil(new_layer_size/deep_ann.hidden_layer_division);
                h = size(weights{layer-1},2);
                w = new_layer_size;
                weights{layer} = (range(2)-range(1)).*rand(h,w)+range(1);
            end

            h = size(weights{num_hidden_node_layers - 1},2);
            w = num_output_nodes;
            weights{num_hidden_node_layers} = (range(2)-range(1)).*rand(h, w)+range(1);

            %{
            if deep_ann.should_add_bias_to_hidden
                for layer=1:num_hidden_node_layers
                    if mod(layer,2) == 0
                        weights{layer} = [ones(1,size(weights{layer},2));weights{layer}];
                    else
                        weights{layer} = [ones(size(weights{layer},1),1) weights{layer}];
                    end
                end
            end
            %}
            % Matrix to track training error for plotting
            training_accuracy = zeros(deep_ann.training_iters, 2);

            % Track training h values
            training_h = cell(num_hidden_node_layers - 1,1);

            iter = 0;
            while iter < deep_ann.training_iters
                iter = iter + 1;    
                %% Forward Propagation
                % Compute first hidden layer
                training_h{1} = activation_fxn(std_training_fields * weights{1});

                % Compute internal hidden layers
                for layer=2:num_hidden_node_layers - 1
                    training_h{layer} = activation_fxn(training_h{layer-1} * weights{layer});
                end

                % Compute output layer
                training_o = activation_fxn(training_h{num_hidden_node_layers - 1} * weights{num_hidden_node_layers});

                %% Backward Propagation
                deltas = cell(num_hidden_node_layers,1);

                % Output layer delta
                deltas{num_hidden_node_layers} = new_training_classes - training_o;

                % Internal hidden layer deltas
                for layer=num_hidden_node_layers-1:-1:1
                    deltas{layer} = (weights{layer + 1} * deltas{layer + 1}')' .* (training_h{layer} .* (1 - training_h{layer}));
                end

                % Update weights
                for layer=num_hidden_node_layers-1:-1:1
                    weights{layer+1} = weights{layer+1} + ((deep_ann.eta/num_training_rows) * deltas{layer+1}' * training_h{layer})';
                end

                % Choose maximum output node as value
                [~,training_o] = max(training_o,[],2);

                % Log training error
                num_correct = numel(find(~(training_classes - training_o)));
                acc = num_correct/num_training_rows;
                training_accuracy(iter,:) = [iter,acc];
            end

            %% Testing
            % Compute first hidden layer
            testing_h = activation_fxn(std_testing_fields * weights{1});

            % Compute internal hidden layers
            for layer=2:num_hidden_node_layers - 1
                testing_h = activation_fxn(testing_h * weights{layer});
            end

            % Compute output layer
            testing_o = activation_fxn(testing_h * weights{num_hidden_node_layers});

            % Choose maximum output node as value
            [~,testing_o] = max(testing_o,[],2);

            % Compute number of correct predictions
            num_correct = numel(find(~(testing_classes - testing_o)));
            testing_accuracy = num_correct/num_testing_rows;

            % Plot the training error
            %{
            figure();
            plot(training_accuracy(:,1), training_accuracy(:,2));
            legend('Training Accuracy');
            xlabel('Iteration');
            ylabel('Accuracy');
            %}
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
                 ANN.binary_to_str(deep_ann.should_add_bias_to_input),...
                 ANN.binary_to_str(deep_ann.should_add_bias_to_hidden),...
                 ANN.binary_to_str(deep_ann.should_std_data),...
                 ANN.binary_to_str(deep_ann.should_perform_PCA),...
                 testing_accuracy);
        end
        function [ s_training_accuracies,s_testing_accuracies ] = cross_validate_ANN( ann,S,classes,fields )
            %rng(0);
            %[num_classes,classes,fields] = load_image_data(image_size,image_size);
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
                
                [testing_accuracy,training_accuracy] = train_ANN(ann,s_training_fields,s_training_classes,s_testing_fields,s_testing_classes);

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
        function s = control_flow_str( deep_ann )
            s1 = ANN.binary_to_str(deep_ann.should_add_bias_to_input);
            s2 = ANN.binary_to_str(deep_ann.should_add_bias_to_hidden);
            s3 = ANN.binary_to_str(deep_ann.should_std_data);
            s4 = ANN.binary_to_str(deep_ann.should_perform_PCA);
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