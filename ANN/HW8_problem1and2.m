clear all;
close all;
rng(0);

% INITIAL VALS
eta = 0.5;
num_hidden_nodes = 20;
num_output_nodes = 1;
activation_fxn = @(x) 1./(1 + exp(-x));
training_iters = 1000;
threshold = 0.5;

%% Load and Standardize Data
% Load the data and randomly permutate
[class, fields] = read_spam_data();

num_data_rows = length(fields(:,1));
num_data_cols = length(fields(1,:));

% Concatinate data for shuffling
fields_and_classes = [fields class];
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

num_training_rows = length(training_fields(:,1));
num_testing_rows = length(testing_fields(:,1));

% Standardize data via training mean and training std dev
[std_training_fields,training_fields_mean,training_fields_std_dev] = standardize_data(training_fields);
std_training_data = [std_training_fields, training_classes];

std_testing_fields = testing_fields - repmat(training_fields_mean,size(testing_fields,1),1);
std_testing_fields = std_testing_fields ./ repmat(training_fields_std_dev,size(std_testing_fields,1),1);
std_testing_data = [std_testing_fields, testing_classes];

% Add bias node and increase column size by 1
std_training_fields = [ones(num_training_rows, 1), std_training_fields];
std_testing_fields = [ones(num_testing_rows, 1), std_testing_fields];
num_data_cols = num_data_cols + 1;

%% Perform Forward/Backward Propagation with Batch Gradient Descent
iter = 0;

% Initialize weights as random 
range = [-1,1];
beta = (range(2)-range(1)).*rand(num_data_cols, num_hidden_nodes) + range(1);
theta = (range(2)-range(1)).*rand(num_hidden_nodes, num_output_nodes) + range(1);

% Matrix to track training error for plotting
training_error = zeros(training_iters, 2);

while iter < training_iters
    iter = iter + 1;    
    %% Forward Propagation
    % Compute hidden layer
    training_h = activation_fxn(std_training_fields * beta);

    % Compute output layer    
    training_o = activation_fxn(training_h * theta);
    
    %% Backward Propagation
    % Compute output error
    delta_output = training_classes - training_o;

    % Update theta
    theta = theta + ((eta/num_training_rows) * delta_output' * training_h)';

    % Compute hidden error
    %delta_hidden = repmat(theta',num_training_rows,1) .* repmat(delta_output,1,num_hidden_nodes) .* (training_h .* (1 - training_h));
    delta_hidden = (theta * delta_output')' .* (training_h .* (1 - training_h));

    % Update beta
    beta = beta + (eta/num_training_rows) * (delta_hidden' * std_training_fields)';

    % Log training error
    % TODO: replace binary_threshold here
    [Accuracy,~,~] = binary_threshold_classifier(threshold, training_o, training_classes);
    err = 1 - Accuracy;
    training_error(iter,:) = [iter,err];
end

%% Testing
% Compute hidden layer
testing_h = activation_fxn(std_testing_fields * beta);

% Compute output layer
testing_o = activation_fxn(testing_h * theta);

% Compute number of correct predictions
[Accuracy,Precision,Recall] = binary_threshold_classifier(threshold, testing_o, testing_classes);

% Compute accuracy as a percentage of classes predicted correctly
fprintf('Accuracy = %f\n', Accuracy);
fprintf('Testing Error = %f\n',1 - Accuracy);

% Plot the training error
figure();
plot(training_error(:,1), training_error(:,2));
legend('Training Error');
xlabel('Iteration');
ylabel('Error');

%% Problem 2
a = 0;
b = 1;
h = 0.1;
n = ceil((a-b)/h);
precision_recall = zeros(n,2);

% Calculate precision and recall varying threshold
iter = 0;
for threshold = a:h:b
    iter = iter + 1;
    [Accuracy, Precision, Recall] = binary_threshold_classifier(threshold, testing_o, testing_classes);
    precision_recall(iter,:) = [Precision, Recall];
end

% Plot PRC
figure();
plot(precision_recall(:,1), precision_recall(:,2), 'o-')
xlabel('Precision');
ylabel('Recall');
