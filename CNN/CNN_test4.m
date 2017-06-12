rng(0);
activation_fxn = @(x) 1./(1 + exp(-x));
eta = 0.5;
training_iters = 1000;
num_filters = 5;

image_size = 28;
filter_size = 9;
conv_size = image_size - filter_size + 1;

%% Image Loading
[num_classes,classes,image_maps] = load_image_data(image_size,image_size);

[std_image_maps,m,s] = standardize_data(image_maps);
%num_training_rows = 1;
num_training_images = size(std_image_maps,3);
training_classes = classes;

% Reformat training classes
new_training_classes = zeros(num_training_images,num_classes);
for i = 1:num_training_images
    new_training_classes(i,training_classes(i)) = 1;
end

%% Variable instantiation
% Single feature map vars
range = [-1,1];
filters = (range(2)-range(1)).*rand(filter_size,filter_size,num_filters) + range(1);
feature_maps = zeros(conv_size,conv_size,num_training_images,num_filters);

% Pooling vars
pool_window_size = 2;
pooled_feature_map = zeros(conv_size/pool_window_size,conv_size/pool_window_size,num_training_images,num_filters);
pool_coords = cell(size(pooled_feature_map,1) * size(pooled_feature_map,2));
pooled_feature_map_max_coords = cell(size(pooled_feature_map,1),size(pooled_feature_map,2),num_filters);

% Shallow ANN vars
num_hidden_nodes = 20;
beta = (range(2)-range(1)).*rand(size(pooled_feature_map,1) * size(pooled_feature_map,2),num_hidden_nodes) + range(1);
theta = (range(2)-range(1)).*rand(num_hidden_nodes,num_classes) + range(1);
training_accuracy = zeros(training_iters, 2);

activated_image_maps = activation_fxn(std_image_maps);

for iter = 1:training_iters
    %% Convolution
    for image = 1:num_training_images
        for filter = 1:num_filters
            feature_maps(:,:,image,filter) = activation_fxn(conv2(std_image_maps(:,:,image),filters(:,:,filter),'valid'));
        end
    end
    
    %% Activating Convolution
    %activated_feature_maps = activation_fxn(feature_maps);

    %% Pooling
    pool_height_start_pt = 1;
    pool_height_end_pt = pool_height_start_pt + pool_window_size - 1;
    for h = 1:(conv_size/pool_window_size)
        pool_width_start_pt = 1;
        pool_width_end_pt = pool_width_start_pt + pool_window_size - 1;

        for w = 1:(conv_size/pool_window_size)
            for filter = 1:num_filters
                pool = feature_maps(pool_height_start_pt:pool_height_end_pt,pool_width_start_pt:pool_width_end_pt,:,filter);
                [x,y,~] = size(pool);
                [pooled_feature_map(h,w,:,filter),pool_max_idx] = max(reshape(pool(:),x*y,[]));
                
                % Track the position of the max value for error propogation
                [max_y,max_x] = ind2sub([x y],pool_max_idx);
                pooled_feature_map_max_coords{h,w,filter} = struct('x',pool_width_start_pt+max_x-1,'y',pool_height_start_pt+max_y-1);
            end
            
            pool_width_start_pt = pool_width_start_pt + pool_window_size;
            pool_width_end_pt = pool_width_end_pt + pool_window_size;
        end

        pool_height_start_pt = pool_height_start_pt + pool_window_size;
        pool_height_end_pt = pool_height_end_pt + pool_window_size;
    end

    %% Normalization
    %Change all negative values to 0
    %pooled_feature_map(pooled_feature_map < 0) = 0;

    %% Fully Connected Layer
    % Flatten pools into vector
    %input_pool_vec = reshape(pooled_feature_map,num_training_images*num_filters,numel(pooled_feature_map(:,:,1,1)));
    input_pool_vec = reshape(pooled_feature_map,numel(pooled_feature_map(:,:,1,1)),num_training_images * num_filters)';

    % Compute hidden layer
    training_h = activation_fxn(input_pool_vec * beta);

    % Compute output layer    
    training_o = activation_fxn(training_h * theta);

    %% Backward Propagation: Shallow ANN
    % Compute output error
    %TODO: REPMAT OUTSIDE OF LOOP
    %delta_output = new_training_classes - training_o;
    delta_output = repmat(new_training_classes,num_filters,1) - training_o;

    % Update theta
    theta = theta + ((eta/num_training_images) * delta_output' * training_h)';

    % Compute hidden error
    delta_hidden = (theta * delta_output')' .* (training_h .* (1 - training_h));

    % Update beta
    beta = beta + (eta/num_training_images) * (delta_hidden' * input_pool_vec)';

    %% Backward Propagation: Pooling and feature map
    % Compute pooling error
    delta_pool = (beta * delta_hidden')' .* (input_pool_vec .* (1 - input_pool_vec));
    [x,y,z,alpha] = size(pooled_feature_map);
    delta_pool = reshape(delta_pool,x,y,z,alpha);

    delta_feature = zeros(size(feature_maps));
    % delta_feature_test = zeros(size(feature_maps));
    for h = 1:size(pooled_feature_map_max_coords,1)
        for w = 1:1:size(pooled_feature_map_max_coords,2)
            x = pooled_feature_map_max_coords{h,w}.x;
            y = pooled_feature_map_max_coords{h,w}.y;
            for layer = 1:num_training_images
                delta_feature(y(layer),x(layer),layer,:) = delta_pool(h,w,layer,:);
%                 for filter = 1:num_filters
%                     delta_feature_test(y(layer),x(layer),layer,filter) = delta_pool(h,w,layer,filter);
%                 end
            end
        end
    end

    % TODO: This is blowing up because of the zeros from the max pooling
    % May not be an error, but something to consider
    % Compute error at convolution
    delta_conv = delta_feature .* (feature_maps .* (1 - feature_maps)); 
            
    %conv2(delta_feature,filters(:,:,1).*(feature_maps(:,:,1,1) .* (1 - feature_maps(:,:,1,1))));
    %conv2(filters(:,:,1),(feature_maps(:,:,1,1) .* (1 - feature_maps(:,:,1,1))),'same')
    
    % Compute gradient at filter
    rot_delta_conv = rot90(delta_conv,2);
    delta_filter = zeros(filter_size,filter_size,num_training_images,num_filters);
    for image = 1:num_training_images
        for filter = 1:num_filters
            %delta_filter(:,:,image,filter) = conv2(std_image_maps(:,:,image),rot_delta_conv(:,:,image,filter),'valid'); %.* (filters(:,:,filter) .* (1 - filter(:,:,filter)));
            delta_filter(:,:,image,filter) = rot90(conv2(std_image_maps(:,:,image),rot_delta_conv(:,:,image,filter),'valid'),2);
            %delta_filter(:,:,image,filter) = conv2(filters(:,:,filter),(feature_maps(:,:,image,filter) .* (1 - feature_maps(:,:,image,filter))),'same');
        end
    end
    
    %size(conv2(delta_conv(:,:,1,1),filters(:,:,1),'valid'))
    %conv2(delta_conv(:,:,1,1)
    
    % Update filter
    % filter = filter + (eta/num_training_images) * mean(delta_filter,3);
    filters = filters + (eta/num_training_images) * squeeze(sum(delta_filter,3));

    %% Training error tracking
    % Choose maximum output node as value
    [~,training_o] = max(training_o,[],2);

    % Log training error
    %num_correct = numel(find(~(training_classes - training_o)));
    num_correct = numel(find(~(repmat(training_classes,num_filters,1) - training_o)));
    acc = num_correct/(num_training_images*num_filters);
    training_accuracy(iter,:) = [iter,acc];
end

%% Plot
plot(training_accuracy(:,1),training_accuracy(:,2))
fprintf("Final Training Accuracy: %f\n",acc);

[a,b] = findIntegerFactorsCloseToSquarRoot(num_filters);
figure();
for filter = 1:num_filters
    m = mean2(filters(:,:,filter));
    s = std2(filters(:,:,filter));
    s(s == 0) = 1;
    std_filter = filters(:,:,filter) - m;
    std_filter = std_filter ./ s;
    
    subplot(a,b,filter);
    imshow(std_filter);
end
