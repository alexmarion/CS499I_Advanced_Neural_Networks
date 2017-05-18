rng(0);
image_size = 40;
[num_classes,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size);

rng(0);
ann = ANN;
[~,~,validation_accuracy,testing_accuracy] = train_ANN(ann,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes)

rng(0);


%% Parameters
deep_ann.num_classes = 15;
deep_ann.hidden_layer_division = 1.29;%1.61803399;%1.4;
deep_ann.training_iters = 1000;
deep_ann.eta = 0.5;
deep_ann.percent_field_retention = 0.95;

%% Control Flow Values
deep_ann.should_add_bias_to_input = true;
deep_ann.should_add_bias_to_hidden = false;
deep_ann.should_std_data = true;
deep_ann.should_perform_PCA = true;

%% Set Initial Vals
num_output_nodes = deep_ann.num_classes;
activation_fxn = @(x) 1./(1 + exp(-x));

% Get number of training rows and number of testing rows
num_training_rows = length(training_fields(:,1));
num_testing_rows = length(testing_fields(:,1));

num_data_cols = length(training_fields(1,:));

%% Perform PCA
if deep_ann.should_perform_PCA
    projection_vectors = PCA(training_fields,deep_ann.percent_field_retention);
    training_fields = training_fields * projection_vectors;
    testing_fields = testing_fields * projection_vectors;

    num_data_cols = length(training_fields(1,:));
end

%% Compute number of hidden layers based on input size and division ratio
num_hidden_node_layers = 0;
num_nodes_in_layer = num_data_cols;
for i = 1:num_testing_rows
    num_nodes_in_layer = ceil(num_nodes_in_layer/deep_ann.hidden_layer_division);
    if num_nodes_in_layer < num_output_nodes
        break
    end
    num_hidden_node_layers = num_hidden_node_layers + 1;
end

% Must have at least 2 hidden layers to be a deep network
if num_hidden_node_layers < 1
    num_hidden_node_layers = 1;
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
weights = cell(num_hidden_node_layers + 1,1);
new_layer_size = ceil(num_data_cols/deep_ann.hidden_layer_division);
weights{1} = (range(2)-range(1)).*rand(num_data_cols, new_layer_size)+range(1);

for layer = 2:num_hidden_node_layers 
    new_layer_size = ceil(new_layer_size/deep_ann.hidden_layer_division);
    h = size(weights{layer-1},2);
    w = new_layer_size;
    weights{layer} = (range(2)-range(1)).*rand(h,w)+range(1);
end

h = size(weights{num_hidden_node_layers},2);
w = num_output_nodes;
weights{num_hidden_node_layers + 1} = (range(2)-range(1)).*rand(h, w)+range(1);

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
training_out = cell(num_hidden_node_layers,1);

iter = 0;
while iter < deep_ann.training_iters
    iter = iter + 1;    
    %% Forward Propagation
    % Compute first hidden layer
    training_out{1} = activation_fxn(std_training_fields * weights{1});
    
    % Compute internal hidden layers
    for layer=2:num_hidden_node_layers + 1
        training_out{layer} = activation_fxn(training_out{layer-1} * weights{layer});
    end
    
    % Compute output layer
    %training_o = activation_fxn(training_out{num_hidden_node_layers} * weights{num_hidden_node_layers + 1});

    %% Backward Propagation
    deltas = cell(num_hidden_node_layers + 1,1);
    
    % Output layer delta
    deltas{num_hidden_node_layers+1} = new_training_classes - training_out{end};
    weights{end} = weights{end} + ((deep_ann.eta/num_training_rows) * training_out{end-1}' * deltas{end});

    % Update weights and compute internal hidden layer deltas
    for layer=num_hidden_node_layers:-1:2
        deltas{layer} = (deltas{layer + 1} * weights{layer + 1}') .* (training_out{layer} .* (1 - training_out{layer}));                
        weights{layer} = weights{layer} + ((deep_ann.eta/num_training_rows) * deltas{layer}' * training_out{layer-1})';
    end
    
    % Input layer delta
    deltas{1} = (deltas{2} * weights{2}') .* (training_out{1} .* (1 - training_out{1}));
    weights{1} = weights{1} + ((deep_ann.eta/num_training_rows) * deltas{1}' * std_training_fields)';
    
    % Choose maximum output node as value
    [~,training_o] = max(training_out{end},[],2);

    % Log training error
    num_correct = numel(find(~(training_classes - training_o)));
    acc = num_correct/num_training_rows;
    training_accuracy(iter,:) = [iter,acc];
end

%% Testing
% Compute first hidden layer
testing_h = activation_fxn(std_testing_fields * weights{1});

% Compute internal hidden layers
for layer=2:num_hidden_node_layers
    testing_h = activation_fxn(testing_h * weights{layer});
end

% Compute output layer
testing_o = activation_fxn(testing_h * weights{num_hidden_node_layers + 1});

% Choose maximum output node as value
[~,testing_o] = max(testing_o,[],2);

% Compute number of correct predictions
num_correct = numel(find(~(testing_classes - testing_o)));
testing_accuracy = num_correct/num_testing_rows

% Plot the training error
figure();
plot(training_accuracy(:,1), training_accuracy(:,2));
legend('Training Accuracy');
xlabel('Iteration');
ylabel('Accuracy');
