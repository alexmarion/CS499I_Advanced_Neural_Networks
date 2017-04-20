clear all;
close all;
rng(0);

%% Control Flow Values
data_selection_type = 0;

should_add_bias_to_input = false;
should_add_bias_to_hidden = false;
should_std_data = false;
should_perform_PCA = false;
percent_field_retention = .95;

image_height = 40;
image_width = 40;

%% Load Data
% Load the data and randomly permutate
[num_classes, classes, fields] = load_image_data(image_height,image_width);


%% Set Initial Vals
eta = 0.5;
num_hidden_nodes = 20;
num_output_nodes = num_classes;
activation_fxn = @(x) 1./(1 + exp(-x));
training_iters = 1000;

%% Select Training and Testing Sets
% Initialize vars
classifiers = unique(classes);
num_classes = numel(classifiers);
num_data_rows = length(fields(:,1));
num_data_cols = length(fields(1,:));

[training_fields,training_classes,testing_fields,testing_classes] = get_training_and_testing_sets(fields,classes,data_selection_type);

% Get number of training rows and number of testing rows
num_training_rows = length(training_fields(:,1));
num_testing_rows = length(testing_fields(:,1));

%% Standardize Data
if should_std_data
    % Standardize data via training mean and training std dev
    [std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);
    % std_training_data = [std_training_fields, training_classes];

    std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
    std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
    % std_testing_data = [std_testing_fields, testing_classes];
else
    std_training_fields = training_fields;
    std_testing_fields = testing_fields;
end

%% Perform PCA
if should_perform_PCA
    projection_vectors = PCA(std_training_fields,percent_field_retention);
    std_training_fields = std_training_fields * projection_vectors;
    std_testing_fields = std_testing_fields * projection_vectors;

    num_data_cols = length(std_training_fields(1,:));
end

%% Add bias nodes to input layer
if should_add_bias_to_input
    % Add bias node and increase number of columns by 1
    std_training_fields = [ones(num_training_rows, 1), std_training_fields];
    std_testing_fields = [ones(num_testing_rows, 1), std_testing_fields];
    num_data_cols = num_data_cols + 1;
end

% Reformat training classes
new_training_classes = zeros(num_training_rows,num_classes);
for i = 1:num_training_rows
    new_training_classes(i,training_classes(i)) = 1;
end

%% Perform Forward/Backward Propagation with Batch Gradient Descent
iter = 0;

% Initialize weights as random 
range = [-1,1];
beta = (range(2)-range(1)).*rand(num_data_cols, num_hidden_nodes) + range(1);
theta = (range(2)-range(1)).*rand(num_hidden_nodes, num_output_nodes) + range(1);

if should_add_bias_to_hidden
    % theta = [ones(num_hidden_nodes,1) theta];
    % beta = [ones(1,num_hidden_nodes);beta];
    beta = [ones(num_data_cols,1) beta];
    theta = [ones(1,num_output_nodes);theta];
end

% Matrix to track training error for plotting
training_accuracy = zeros(training_iters, 2);

while iter < training_iters
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
    theta = theta + ((eta/num_training_rows) * delta_output' * training_h)';

    % Compute hidden error
    delta_hidden = (theta * delta_output')' .* (training_h .* (1 - training_h));

    % Update beta
    beta = beta + (eta/num_training_rows) * (delta_hidden' * std_training_fields)';

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

% Plot the training error
figure();
plot(training_accuracy(:,1), training_accuracy(:,2));
legend('Training Accuracy');
xlabel('Iteration');
ylabel('Accuracy');