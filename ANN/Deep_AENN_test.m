%% Init Params
rng(0);
image_size = 40;
S = 9;
[~,classes,fields] = load_image_data(image_size,image_size);

% Parameters
deep_ann.num_classes = 15;
deep_ann.hidden_layer_division = 1.5;%1.29%1.61803399
deep_ann.training_iters = 1000;
deep_ann.eta = 1.5;
deep_ann.percent_field_retention = 0.95;

deep_ann.should_add_bias_to_input = true;
deep_ann.should_add_bias_to_hidden = false;
deep_ann.should_std_data = true;
deep_ann.should_perform_PCA = true;

deep_ann.should_plot_train = false;
deep_ann.should_plot_s_folds = false;

%% Set Initial Vals
num_output_nodes = deep_ann.num_classes;
[num_data_rows, num_features] = size(fields);


%             %% Perform PCA
%             if deep_ann.should_perform_PCA
%                 projection_vectors = PCA(training_fields,deep_ann.percent_field_retention);
%                 training_fields = training_fields * projection_vectors;
%                 validation_fields = validation_fields * projection_vectors;
%                 testing_fields = testing_fields * projection_vectors;
% 
%                 num_data_rows = length(training_fields(1,:));
%             end

%% Compute number of hidden layers based on input size and division ratio
deep_ann.num_hidden_node_layers = 0;
num_nodes_in_layer = num_data_rows;
for i = 1:num_data_rows
    num_nodes_in_layer = ceil(num_nodes_in_layer/deep_ann.hidden_layer_division);
    if num_nodes_in_layer < num_output_nodes
        break
    end
    deep_ann.num_hidden_node_layers = deep_ann.num_hidden_node_layers + 1;
end

% Must have at least 2 hidden layers to be a deep network
if deep_ann.num_hidden_node_layers < 1
    deep_ann.num_hidden_node_layers = 1;
end

%% Standardize Data
if deep_ann.should_std_data
    % Standardize data via training mean and training std dev
    [std_fields,~,~] = standardize_data(fields);
else
    std_fields = training_fields;
end

% Reformat training classes
new_training_classes = zeros(num_data_rows,deep_ann.num_classes);
for i = 1:num_data_rows
    new_training_classes(i,classes(i)) = 1;
end

%% Perform Forward/Backward Propagation with Batch Gradient Descent
% Initialize weights as random    
range = [-1,1];
deep_ann.weights = cell(deep_ann.num_hidden_node_layers + 1,1);
new_layer_size = ceil(num_features/deep_ann.hidden_layer_division);
deep_ann.weights{1} = (range(2)-range(1)).*rand(num_features, new_layer_size)+range(1);

for layer = 2:deep_ann.num_hidden_node_layers 
    new_layer_size = ceil(new_layer_size/deep_ann.hidden_layer_division);
    h = size(deep_ann.weights{layer-1},2);
    w = new_layer_size;
    deep_ann.weights{layer} = (range(2)-range(1)).*rand(h,w)+range(1);
end

h = size(deep_ann.weights{deep_ann.num_hidden_node_layers},2);
w = num_output_nodes;
deep_ann.weights{deep_ann.num_hidden_node_layers + 1} = (range(2)-range(1)).*rand(h, w)+range(1);

if deep_ann.should_add_bias_to_hidden
    for layer=1:deep_ann.num_hidden_node_layers + 1
        if mod(layer,2) == 0
            deep_ann.weights{layer} = [ones(1,size(deep_ann.weights{layer},2));deep_ann.weights{layer}];
        else
            deep_ann.weights{layer} = [ones(size(deep_ann.weights{layer},1),1) deep_ann.weights{layer}];
        end
    end
end

% Sanity check
if size(deep_ann.weights{end}, 2) > num_output_nodes
    deep_ann.weights{end} = deep_ann.weights{end}(:,1:num_output_nodes);
end

% Matrix to track training error for plotting
training_accuracy = zeros(deep_ann.training_iters, 2);

% Track training h values
training_out = cell(deep_ann.num_hidden_node_layers + 1,1);
training_out{1} = std_fields;

for layer=2:deep_ann.num_hidden_node_layers
%     if (layer == 2)
    deep_ann.activation_fxn = @(x) 1./(1 + exp(-x));
%     else
%         deep_ann.activation_fxn = @(x) x;
%     end
%     
    iter = 0;
    input_fields = training_out{layer-1};
    output_weights = (range(2)-range(1)).*rand(length(deep_ann.weights{layer}), length(input_fields))+range(1);
    while iter < deep_ann.training_iters
        iter = iter + 1;

        %% Forward Propagation
        training_out{layer} = deep_ann.activation_fxn(input_fields * deep_ann.weights{layer - 1});
        false_output = deep_ann.activation_fxn(training_out{layer} * output_weights);

        %% Backward Propagation
        delta_output = input_fields - false_output;
        output_weights = output_weights + ((deep_ann.eta/num_data_rows) * delta_output' * training_out{layer})';

        delta_hidden = (deep_ann.weights{layer - 1}' * delta_output')' .* (training_out{layer} .* (1 - training_out{layer}));
        deep_ann.weights{layer - 1} = deep_ann.weights{layer - 1} + (deep_ann.eta/num_data_rows) * (delta_hidden' * training_out{layer -1})';

    end
end

%% softmax

X = training_out{6};

softmax(X);

% W = 0.01 * randn(size(final_layer));
% b = zeros(1,num_data_rows);
% 
% step_size = 1;
% reg = .001;
% 
% 
% for i=0:100
%   
%   scores = dot(X, W) + b' ;
%   
%   exp_scores = exp(scores);
%   probs = exp_scores / sum(exp_scores);
%   
%   corect_logprobs = log(probs[range(num_examples),y]);
%   data_loss = sum(corect_logprobs)/num_examples;
%   reg_loss = 0.5*reg*sum(W*W);
%   loss = data_loss + reg_loss;
%   
%   if mod(i, 10) == 0
%     print "iteration %d: loss %f" % (i, loss)
%   end
  
%   dscores = probs
%   dscores[range(num_examples),y] -= 1
%   dscores /= num_examples
%   
%   dW = np.dot(X.T, dscores)
%   db = np.sum(dscores, axis=0, keepdims=True)
%   
%   dW += reg*W # regularization gradient
%   
%   # perform a parameter update
%   W += -step_size * dW
%   b += -step_size * db
% end