rng(0);
activation_fxn = @(x) 1./(1 + exp(-x));
eta = 0.5;
training_iters = 1000;
num_filters = 1;

conv_type = 'valid';

image_size = 28;
filter_size = 9;%ceil(image_size * (1/3));
conv_size = image_size - filter_size + 1;
%conv_size = size(conv2(zeros(image_size),zeros(filter_size)), 1);


if mod(conv_size,2) ~= 0
    error("Image size - filter size + 1 must be divisble by 2\n")
end


%% Image Loading
[num_classes,classes,image_maps] = load_image_data(image_size,image_size);
[training_maps,training_classes,validation_maps,validation_classes,testing_maps,testing_classes] = get_training_and_testing_sets(image_maps,classes);

% Standardize data
[std_training_maps,m,s] = standardize_data(training_maps);

std_validation_maps = validation_maps - m;
std_validation_maps = std_validation_maps ./ s;

std_testing_maps = testing_maps - m;
std_testing_maps = std_testing_maps ./ s;

num_training_images = size(std_training_maps,3);

% Reformat training classes
new_training_classes = zeros(num_training_images,num_classes);
for i = 1:num_training_images
    new_training_classes(i,training_classes(i)) = 1;
end

%% Variable instantiation
% Feature map vars
range = [-1,1];
filters = (range(2)-range(1)).*rand(filter_size,filter_size,num_filters) + range(1);
feature_maps = zeros(conv_size,conv_size,num_training_images,num_filters);

% Pooling vars
pool_window_size = 2;
pooled_feature_map = zeros(conv_size/pool_window_size,conv_size/pool_window_size,num_training_images,num_filters);
pool_coords = cell(size(pooled_feature_map,1) * size(pooled_feature_map,2));
pooled_feature_map_max_coords = cell(size(pooled_feature_map,1),size(pooled_feature_map,2),num_filters);

% Shallow ANN vars
num_hidden_nodes = 30;
beta = (range(2)-range(1)).*rand(size(pooled_feature_map,1) * size(pooled_feature_map,2),num_hidden_nodes,num_filters) + range(1);
theta = (range(2)-range(1)).*rand(num_hidden_nodes,num_classes,num_filters) + range(1);
training_accuracy = zeros(training_iters, 2);

for iter = 1:training_iters
    %% Convolution
    for image = 1:num_training_images
        for filter = 1:num_filters
            feature_maps(:,:,image,filter) = activation_fxn(conv2(std_training_maps(:,:,image),filters(:,:,filter),conv_type));
        end
    end

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

    %% Fully Connected Layer
    % Flatten pools into vector
    input_pool_vec = reshape(pooled_feature_map,num_training_images,numel(pooled_feature_map(:,:,1,1)),num_filters);
    %input_pool_vec = reshape(pooled_feature_map,numel(pooled_feature_map(:,:,1,1)),num_training_images,num_filters);
    %input_pool_vec = permute(input_pool_vec, [2,1,3,4]);
    
    num_correct = 0;
    for filter = 1:num_filters
        % Compute hidden layer
        training_h = activation_fxn(input_pool_vec(:,:,filter) * beta(:,:,filter));
        
        % Compute output layer    
        training_o = activation_fxn(training_h * theta(:,:,filter));
        
        %% Backward Propagation: Shallow ANN
        % Compute output error
        delta_output = new_training_classes - training_o;

        % Update theta
        theta(:,:,filter) = theta(:,:,filter) + ((eta/num_training_images) * delta_output' * training_h)';

        % Compute hidden error
        delta_hidden = (theta(:,:,filter) * delta_output')' .* (training_h .* (1 - training_h));

        % Update beta
        beta(:,:,filter) = beta(:,:,filter) + (eta/num_training_images) * (delta_hidden' * input_pool_vec(:,:,filter))';
    
        %% Backward Propagation: Pooling and feature map
        % Compute pooling error
        delta_pool = (beta(:,:,filter) * delta_hidden')' .* (input_pool_vec(:,:,filter) .* (1 - input_pool_vec(:,:,filter)));
        [x,y,z,alpha] = size(pooled_feature_map);
        delta_pool = reshape(delta_pool,x,y,z);
        
        
        delta_feature = zeros(size(feature_maps(:,:,:,filter)));
        for image = 1:num_training_images
            delta_feature(:,:,image) = kron(delta_pool(:,:,image), ones(pool_window_size));
        end
        
        
%         delta_feature = zeros(size(feature_maps(:,:,:,filter)));
%         for h = 1:size(pooled_feature_map_max_coords,1)
%             for w = 1:1:size(pooled_feature_map_max_coords,2)
%                 x = pooled_feature_map_max_coords{h,w,filter}.x;
%                 y = pooled_feature_map_max_coords{h,w,filter}.y;
%                 for layer = 1:num_training_images
%                     delta_feature(y(layer),x(layer),layer) = delta_pool(h,w,layer);
%                 end
%             end
%         end
        
        % Compute error at convolution
        delta_conv = delta_feature .* (feature_maps(:,:,:,filter) .* (1 - feature_maps(:,:,:,filter)));
        
        % Compute gradient at filter
        rot_delta_conv = rot90(delta_conv,2);
        delta_filter = zeros(filter_size,filter_size,num_training_images);
        conv_grad = zeros(filter_size,filter_size,num_training_images);
        for image = 1:num_training_images 
            %delta_filter(:,:,image) = conv2(std_image_maps(:,:,image),rot_delta_conv(:,:,image),conv_type);
            %delta_filter(:,:,image) = rot90(conv2(std_training_maps(:,:,image),rot_delta_conv(:,:,image),conv_type),2);
            delta_filter(:,:,image) = rot90(conv2(std_training_maps(:,:,image),delta_conv(:,:,image),conv_type),2);
            
            % Update filter
            %filters(:,:,filter) = filters(:,:,filter) + (eta/(num_training_images*num_filters)) * (squeeze(sum(delta_filter,3)) * conv2(std_training_maps(:,:,image),ones(20),conv_type));
            conv_grad(:,:,image) = conv2(delta_filter(:,:,image),activation_fxn(rot90(std_training_maps(:,:,image),2)),'same');
        end

        % Update filter
        conv_grad = squeeze(sum(conv_grad,3));
        filters(:,:,filter) = filters(:,:,filter) + (eta/(num_training_images*num_filters)) * conv_grad;

        %% Training error tracking
        % Choose maximum output node as value
        [~,training_o] = max(training_o,[],2);
        
        % Log training error
        num_correct = num_correct + numel(find(~(training_classes - training_o)));
    end
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

%% Testing Accuracy
num_testing_images = size(std_testing_maps,3);

% Feature map vars
feature_maps = zeros(conv_size,conv_size,num_testing_images,num_filters);

% Pool vars
pooled_feature_map = zeros(conv_size/pool_window_size,conv_size/pool_window_size,num_testing_images,num_filters);
pool_coords = cell(size(pooled_feature_map,1) * size(pooled_feature_map,2));
pooled_feature_map_max_coords = cell(size(pooled_feature_map,1),size(pooled_feature_map,2),num_filters);

%% Convolution
for image = 1:num_testing_images
    for filter = 1:num_filters
        feature_maps(:,:,image,filter) = activation_fxn(conv2(std_testing_maps(:,:,image),filters(:,:,filter),conv_type));
    end
end

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
        end

        pool_width_start_pt = pool_width_start_pt + pool_window_size;
        pool_width_end_pt = pool_width_end_pt + pool_window_size;
    end

    pool_height_start_pt = pool_height_start_pt + pool_window_size;
    pool_height_end_pt = pool_height_end_pt + pool_window_size;
end

%% Fully Connected Layer
% Flatten pools into vector
input_pool_vec = reshape(pooled_feature_map,num_testing_images,numel(pooled_feature_map(:,:,1,1)),num_filters);

num_correct = 0;
for filter = 1:num_filters
    % Compute hidden layer
    testing_h = activation_fxn(input_pool_vec(:,:,filter) * beta(:,:,filter));

    % Compute output layer    
    testing_o = activation_fxn(testing_h * theta(:,:,filter));

    %% Testing error tracking
    % Choose maximum output node as value
    [~,testing_o] = max(testing_o,[],2);

    % Log training error
    num_correct = num_correct + numel(find(~(testing_classes - testing_o)));
end
acc = num_correct/(num_testing_images*num_filters);
fprintf("Testing Accuracy: %f\n",acc);
