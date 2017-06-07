rng(0);
activation_fxn = @(x) 1./(1 + exp(-x));
eta = 0.5;
training_iters = 1000;
        
image_height = 100;
image_width = 100;
stride = 25;
window_size = 50;

% image_height = 10;
% image_width = 10;
% stride = 1;
% window_size = 5;

horizontal_stride_test = 1:stride:image_width - window_size + 1;
vertical_stride_test = 1:stride:image_height -  window_size + 1;

num_horizontal_windows = length(horizontal_stride_test);
num_vertical_windows = length(vertical_stride_test);

if horizontal_stride_test(end) + window_size - 1 ~= image_width
    error('Stride and Window size do not fit image (horizontal)')
end
if vertical_stride_test(end) + window_size - 1 ~= image_width
    error('Stride and Window size do not fit image (vertical)')
end

%% Image Loading
[num_classes,classes,image_maps] = load_image_data(image_height,image_width);

[std_image_maps,m,s] = standardize_data(image_maps);
num_training_rows = size(std_image_maps,3);
training_classes = classes;

% Reformat training classes
new_training_classes = zeros(num_training_rows,num_classes);
for i = 1:num_training_rows
    new_training_classes(i,training_classes(i)) = 1;
end

%% Get coordinates for subimages
window_coords = cell(num_vertical_windows * num_horizontal_windows, 1);
window_count = 1;

% Slightly different for first iteration (because of indexing from 1)
height_start_pt = 1;
height_end_pt = height_start_pt + window_size - 1;
for h = 1:num_vertical_windows
    width_start_pt = 1;
    width_end_pt = width_start_pt + window_size - 1;
    for w = 1:num_horizontal_windows

        % Track x and y of window coordinates
        window_coords{window_count} = struct('x1',width_start_pt,'x2',width_end_pt, ...
                                            'y1',height_start_pt,'y2',height_end_pt);
        window_count = window_count + 1;
        
        width_start_pt = width_start_pt + stride;
        width_end_pt = width_start_pt + window_size - 1;
    end

    height_start_pt = height_start_pt + stride;
    height_end_pt = height_start_pt + window_size - 1;
end

%% Simple error checking
for w = 1:length(window_coords)
    wc = window_coords{w};
    if length(wc.x1:wc.x2) ~= window_size
        error('help')
    end
    if length(wc.y1:wc.y2) ~= window_size
        error('help')
    end
end

%% Variable instantiation
% Single feature map vars
range = [-1,1];
filter = (range(2)-range(1)).*rand(window_size, window_size) + range(1);
feature_map = zeros(image_height,image_width,num_training_rows);

% Pooling vars
pool_window_size = 2;
pooled_feature_map = zeros(image_height/pool_window_size, image_width/pool_window_size,num_training_rows);
pooled_feature_map_max_coords = cell(size(pooled_feature_map,1),size(pooled_feature_map,2));

% Shallow ANN vars
num_hidden_nodes = 20;
beta = (range(2)-range(1)).*rand(size(pooled_feature_map,1) * size(pooled_feature_map,2),num_hidden_nodes) + range(1);
theta = (range(2)-range(1)).*rand(num_hidden_nodes,num_classes) + range(1);
training_accuracy = zeros(training_iters, 2);

for iter = 1:training_iters
    %% Convolution/RELU
    for w = 1:length(window_coords)
        wc = window_coords{w};
        window = std_image_maps(wc.y1:wc.y2,wc.x1:wc.x2,:);
        feature_map(wc.y1:wc.y2,wc.x1:wc.x2,:) = activation_fxn(filter .* window);
    end
    
    %% Pooling
    pool_height_start_pt = 1;
    pool_height_end_pt = pool_height_start_pt + pool_window_size - 1;
    for h = 1:(image_height/pool_window_size)
        pool_width_start_pt = 1;
        pool_width_end_pt = pool_width_start_pt + pool_window_size - 1;

        for w = 1:(image_width/pool_window_size)
            %pool = feature_map(pool_height_start_pt:pool_height_end_pt,pool_width_start_pt:pool_width_end_pt);
            pool = feature_map(pool_height_start_pt:pool_height_end_pt,pool_width_start_pt:pool_width_end_pt,:);
            [x,y,~] = size(pool);
            %[pooled_feature_map(h,w),pool_max_idx] = max(pool(:));
            [pooled_feature_map(h,w,:),pool_max_idx] = max(reshape(pool(:),x*y,[]));
            
            % Track the position of the max value for error propogation
            %[max_y,max_x] = ind2sub(size(pool),pool_max_idx);
            [max_y,max_x] = ind2sub([x y],pool_max_idx);
            pooled_feature_map_max_coords{h,w} = struct('x',pool_width_start_pt+max_x-1,'y',pool_height_start_pt+max_y-1);

            pool_width_start_pt = pool_width_start_pt + pool_window_size;
            pool_width_end_pt = pool_width_end_pt + pool_window_size;
        end

        pool_height_start_pt = pool_height_start_pt + pool_window_size;
        pool_height_end_pt = pool_height_end_pt + pool_window_size;
    end
    
    %% Normalization
    %Change all negative values to 0
    
    %% Fully Connected Layer
    % Flatten pools into vector
    %input_pool_vec = reshape(pooled_feature_map,1,numel(pooled_feature_map));
    input_pool_vec = reshape(pooled_feature_map,num_training_rows,numel(pooled_feature_map(:,:,1)));
    
    % Compute hidden layer
    training_h = activation_fxn(input_pool_vec * beta);

    % Compute output layer    
    training_o = activation_fxn(training_h * theta);
    
    %% Backward Propagation: Shallow ANN
    % Compute output error
    delta_output = new_training_classes - training_o;

    % Update theta
    theta = theta + ((eta/num_training_rows) * delta_output' * training_h)';

    % Compute hidden error
    delta_hidden = (theta * delta_output')' .* (training_h .* (1 - training_h));

    % Update beta
    beta = beta + (eta/num_training_rows) * (delta_hidden' * input_pool_vec)';
    
    %% Backward Propagation: Pooling and feature map
    % Compute pooling error
    delta_pool = (beta * delta_hidden')' .* (input_pool_vec .* (1 - input_pool_vec));
    [x,y,z] = size(pooled_feature_map);
    delta_pool = reshape(delta_pool,x,y,z);

    % Flip feature map 180 deg so that all filters are flipped
    feature_map = rot90(rot90(feature_map));

    % Compute feature error
    delta_feature = zeros(size(feature_map));
    for h = 1:size(pooled_feature_map_max_coords,1)
        for w = 1:1:size(pooled_feature_map_max_coords,2)
            x = pooled_feature_map_max_coords{h,w}.x;
            y = pooled_feature_map_max_coords{h,w}.y;
            % TODO: Figure out the vectorized version of this, wasnt
            % working with below:
            % delta_feature(y,x,:) = delta_pool(h,w,:);
            for layer = 1:num_training_rows
                delta_feature(y(layer),x(layer),layer) = delta_pool(h,w,layer);
            end
        end
    end

    % Update feature map
    % TODO: Is the std std_image_maps wrong?
    delta_feature_T = permute(delta_feature,[2 1 3]);
    feature_map = feature_map + (eta/num_training_rows) * permute((delta_feature_T .* std_image_maps),[2 1 3]);

    %% Backward Propagation: Filter
    % Update filter
    delta_filter = 0;
    for w = 1:length(window_coords)
        wc = window_coords{w};
        window = std_image_maps(wc.y1:wc.y2,wc.x1:wc.x2,1);
        feature_map_window = feature_map(wc.y1:wc.y2,wc.x1:wc.x2);

        %delta_pool = (beta * delta_hidden')' .* (input_pool_vec .* (1 - input_pool_vec));
        delta_pool_T = permute(delta_pool,[2 1 3]);
        %delta_filter = delta_filter + permute((feature_map_window .* delta_pool_T),[2 1 3]) .* (filter .* (1 - filter));
        delta_filter = delta_filter + mean(permute((feature_map_window .* delta_pool_T),[2 1 3]),3) .* (filter .* (1 - filter));

        
        %feature_map(wc.y1:wc.y2,wc.x1:wc.x2) = activation_fxn(filter .* window);
    end

    delta_filter = delta_filter / length(window_coords);
    delta_filter_T = permute(delta_filter,[2 1 3]);
    filter = filter + (eta/num_training_rows) * permute((delta_filter_T .* feature_map_window),[2 1 3]);

    % Flip feature map 180 deg back
    feature_map = rot90(rot90(feature_map));
    
    %% Training error tracking
    % Choose maximum output node as value
    [~,training_o] = max(training_o,[],2);

    % Log training error
    num_correct = numel(find(~(training_classes - training_o)));
    acc = num_correct/num_training_rows;
    training_accuracy(iter,:) = [iter,acc];
end

%figure();
%imshow(std_image_maps(:,:,1))