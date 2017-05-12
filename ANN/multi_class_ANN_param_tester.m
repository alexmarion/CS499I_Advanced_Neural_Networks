%% Load Data
rng(0);
image_size = 40;
S = 9;
[~,classes,fields] = load_image_data(image_size,image_size);
[num_classes,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size);

% Perform PCA
percent_field_retention = 0.95;
projection_vectors = PCA(training_fields,percent_field_retention);
pca_training_fields = training_fields * projection_vectors;
pca_testing_fields = testing_fields * projection_vectors;

% Need this for number of hidden nodes testing
num_pca_data_cols = size(pca_training_fields,2);

%{
ann = ANN;
ann.should_plot_train = true;
%[~,training_accuracy,validation_accuracy,testing_accuracy] = train_ANN(ann,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes);
ann.should_plot_train = false;
ann.should_plot_s_folds = true;
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
%}
%% Image Size and PCA Retention Rate Testing
image_size_start_pt = 10;
image_size_end_pt = 100;
image_sizes = image_size_start_pt:5:image_size_end_pt;

percent_ret_start_pt = 0.10;
percent_ret_end_pt = 1;
percent_field_retentions = percent_ret_start_pt:0.05:percent_ret_end_pt;

training_accuracies = zeros(numel(image_sizes) * numel(percent_field_retentions),S + 2);
validation_accuracies = zeros(numel(image_sizes) * numel(percent_field_retentions),S + 2);
testing_accuracies = zeros(numel(image_sizes) * numel(percent_field_retentions),S + 2);

num_fields_testing_ANN = ANN;
% could use ((is - 1) * numel(percent_field_retentions)) + pfr but seems
% more efficient to just count
idx_count = 1;
for is=1:numel(image_sizes)
    img_size = image_sizes(is);
    
    % Load images with desired size
    [~,img_size_classes,img_size_fields] = load_image_data(img_size,img_size);
    
    fprintf('Image Size: %d\n',img_size);
    for pfr=1:numel(percent_field_retentions)
        percent_field_retention = percent_field_retentions(pfr);
        num_fields_testing_ANN.percent_field_retention = percent_field_retention;
        
        fprintf('PFR: %f\n',percent_field_retention);
        
        rng(0);
        [~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(num_fields_testing_ANN,S,img_size_fields,img_size_classes);
        training_accuracies(idx_count,:) = [img_size,percent_field_retention,s_training_accuracies(:,end)'];
        validation_accuracies(idx_count,:) = [img_size,percent_field_retention,s_validation_accuracies(:,2)'];
        testing_accuracies(idx_count,:) = [img_size,percent_field_retention,s_testing_accuracies(:,2)'];
        
        idx_count = idx_count + 1;
    end
end

[xx,yy] = meshgrid(image_sizes,percent_field_retentions);

training_fig = figure();
mean_training_acc = mean(training_accuracies(:,3:S+2),2);
training_acc_matrix = reshape(mean_training_acc,numel(percent_field_retentions),numel(image_sizes));
surf(xx,yy,training_acc_matrix,'FaceAlpha',0.9);
xlabel('Image Size');
ylabel('Percent Field Retention');
zlabel('Accuracy');
title('Training Set');
colorbar;

validation_fig = figure();
mean_validation_acc = mean(validation_accuracies(:,3:S+2),2);
validation_acc_matrix = reshape(mean_validation_acc,numel(percent_field_retentions),numel(image_sizes));
surf(xx,yy,validation_acc_matrix,'FaceAlpha',0.9);
xlabel('Image Size');
ylabel('Percent Field Retention');
zlabel('Accuracy');
title('Validation Set');
colorbar;

testing_fig = figure();
mean_testing_acc = mean(testing_accuracies(:,3:S+2),2);
testing_acc_matrix = reshape(mean_testing_acc,numel(percent_field_retentions),numel(image_sizes));
surf(xx,yy,testing_acc_matrix,'FaceAlpha',0.9);
xlabel('Image Size');
ylabel('Percent Field Retention');
zlabel('Accuracy');
title('Testing Set');
colorbar;

% Save image and data
saveas(training_fig,'../Latex/figs/num_fields_empirical_training.png');
saveas(validation_fig,'../Latex/figs/num_fields_empirical_validation.png');
saveas(testing_fig,'../Latex/figs/num_fields_empirical_testing.png');
save('../Data/num_fields_empirical','training_accuracies','validation_accuracies','testing_accuracies');

%% Iteration Testing
start_pt = 1;
end_pt = 2000;
training_iters = start_pt:10:end_pt;
training_accuracies = zeros(numel(training_iters),S+1);
validation_accuracies = zeros(numel(training_iters),S+1);
testing_accuracies = zeros(numel(training_iters),S+1);

iter_test_ANN = ANN;

for i=1:numel(training_iters)
    num_iters = training_iters(i);
    disp(num_iters);
    iter_test_ANN.training_iters = num_iters;

    rng(0);
    [~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(iter_test_ANN,S,fields,classes);
    training_accuracies(i,:) = [num_iters,s_training_accuracies(:,end)'];
    validation_accuracies(i,:) = [num_iters,s_validation_accuracies(:,2)'];
    testing_accuracies(i,:) = [num_iters,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(validation_accuracies(:,1), mean(validation_accuracies(:,2:S+1),2),'g');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Avg. Training Accuracy', 'Avg. Validation Accuracy', 'Avg. Testing Accuracy','Location','southwest')
xlabel('Number of Iterations');
ylabel('Accuracy');
hold off;
% Save image and data
saveas(fig,'../Latex/figs/num_iterations_empirical.png');
save('../Data/num_iterations_empirical','training_accuracies','validation_accuracies','testing_accuracies')

%% Number of Hidden Nodes Testing
start_pt = 15;
end_pt = 32;
num_hidden_nodes = start_pt:1:end_pt;
training_accuracies = zeros(numel(num_hidden_nodes),S + 1);
validation_accuracies = zeros(numel(num_hidden_nodes),S+1);
testing_accuracies = zeros(numel(num_hidden_nodes),S + 1);

hidden_nodes_ANN_test = ANN;

for i=1:numel(num_hidden_nodes)
    num_hidden = num_hidden_nodes(i);
    disp(num_hidden);
    hidden_nodes_ANN_test.num_hidden_nodes = num_hidden;

    rng(0);
    [~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(hidden_nodes_ANN_test,S,fields,classes);
    training_accuracies(i,:) = [num_hidden,s_training_accuracies(:,end)'];
    validation_accuracies(i,:) = [num_hidden,s_validation_accuracies(:,2)'];
    testing_accuracies(i,:) = [num_hidden,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(validation_accuracies(:,1), mean(validation_accuracies(:,2:S+1),2),'g');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Avg. Training Accuracy', 'Avg. Validation Accuracy', 'Avg. Testing Accuracy','Location','southwest')
xlabel('Number of Hidden Nodes');
ylabel('Accuracy');
hold off;
% Save image and data
saveas(fig,'../Latex/figs/num_hidden_nodes_empirical.png');
save('../Data/num_hidden_nodes_empirical','training_accuracies','testing_accuracies')

%% Size of Image Testing
start_pt = 10;
end_pt = 100;
image_sizes = start_pt:1:end_pt;
training_accuracies = zeros(numel(image_sizes),S + 1);
testing_accuracies = zeros(numel(image_sizes),S + 1);

image_size_ANN_test = ANN;

for i=1:numel(image_sizes)
    image_size = image_sizes(i);
    disp(image_size);
    
    % Must re-seed every time
%     rng(0);
%     [img_size_num_classes,img_size_training_fields,img_size_training_classes,img_size_testing_fields,img_size_testing_classes] = load_and_shuffle_data(image_size,data_selection_type);
%     [testing_accuracy,training_accuracy] = train_ANN(image_size_ANN_test,img_size_training_fields,img_size_training_classes,img_size_testing_fields,img_size_testing_classes);
%     training_accuracies(i,:) = [image_size,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [image_size,testing_accuracy];

    rng(0);
    [~,img_size_fields,img_size_classes] = load_image_data(image_size,image_size);
    [s_training_accuracies,s_testing_accuracies] = cross_validate_ANN(image_size_ANN_test,S,img_size_classes,img_size_fields);
    training_accuracies(i,:) = [image_size,s_training_accuracies(:,2)'];
    testing_accuracies(i,:) = [image_size,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Training Accuracy','Testing Accuracy','Location','southwest')
xlabel('Image Size');
ylabel('Accuracy');
hold off;
% Save image and data
saveas(fig,'../Latex/figs/image_size_empirical.png');
save('../Data/image_size_empirical','training_accuracies','testing_accuracies')

%% Learning Rate Testing
start_pt = 0.05;
end_pt = 10;
learning_rates = start_pt:0.1:end_pt;
training_accuracies = zeros(numel(learning_rates),S + 1);
validation_accuracies = zeros(numel(learning_rates),S+1);
testing_accuracies = zeros(numel(learning_rates),S + 1);

eta_ANN_test = ANN;

for i=1:numel(learning_rates)
    learning_rate = learning_rates(i);
    disp(learning_rate);
    eta_ANN_test.eta = learning_rate;
    
    rng(0);
    [~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(eta_ANN_test,S,fields,classes);
    training_accuracies(i,:) = [learning_rate,s_training_accuracies(:,end)'];
    validation_accuracies(i,:) = [learning_rate,s_validation_accuracies(:,2)'];
    testing_accuracies(i,:) = [learning_rate,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(validation_accuracies(:,1), mean(validation_accuracies(:,2:S+1),2),'g');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Avg. Training Accuracy', 'Avg. Validation Accuracy', 'Avg. Testing Accuracy','Location','southwest')
xlabel('Learning Rate');
ylabel('Accuracy');
% Save image and data
saveas(fig,'../Latex/figs/learning_rate_empirical.png');
save('../Data/learning_rate_empirical','training_accuracies','testing_accuracies')

%% Percent Field Retention
start_pt = 0.01;
end_pt = 1;
percent_field_retention = start_pt:0.01:end_pt;
training_accuracies = zeros(numel(percent_field_retention),S + 1);
testing_accuracies = zeros(numel(percent_field_retention),S + 1);

field_ret_ANN_test = ANN;

for i=1:numel(percent_field_retention)
    percent_retention = percent_field_retention(i);
    disp(percent_retention);
    field_ret_ANN_test.percent_field_retention = percent_retention;
%     [testing_accuracy,training_accuracy] = train_ANN(field_ret_ANN_test,training_fields,training_classes,testing_fields,testing_classes);
%     training_accuracies(i,:) = [percent_retention,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [percent_retention,testing_accuracy];

    rng(0);
    [s_training_accuracies,s_testing_accuracies] = cross_validate_ANN(field_ret_ANN_test,S,fields,classes);
    training_accuracies(i,:) = [percent_retention,s_training_accuracies(:,2)'];
    testing_accuracies(i,:) = [percent_retention,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Training Accuracy','Testing Accuracy','Location','southwest')
xlabel('Percent Field Retention');
ylabel('Accuracy');
hold off;
% Save image and data
saveas(fig,'../Latex/figs/percent_field_retention_empirical.png');
save('../Data/percent_field_retention_empirical','training_accuracies','testing_accuracies');

%% Best Value Test
best = ANN;
best.num_hidden_nodes = 291;
best.training_iters = 1000;
best.eta = 0.15;
best.percent_field_retention = 0.94;

image_size = 7;
S = 2;
[~,classes,fields] = load_image_data(image_size,image_size);

[training_accuracy,testing_accuracy] = cross_validate_ANN(best,S,fields,classes);
disp(testing_accuracy);
figure();
plot(training_accuracy(:,1), training_accuracy(:,2));
legend('Training Error');
xlabel('Iteration');
ylabel('Accuracy');

% testing_accuracies(testing_accuracies(:,2)==max(testing_accuracies(:,2)))
%{
img_size = 7;
learning_rate = 0.15;
num_hidden = 291;
iterations = 1000;
retention = 0.94;

[testing_accuracy,training_accuracy] = train_multi_class_ANN(true,false,true,true, ...
                                                                num_hidden,        ...
                                                                iterations,        ...
                                                                img_size,          ...
                                                                learning_rate,     ...
                                                                retention);
disp(testing_accuracy);
figure();
plot(training_accuracy(:,1), training_accuracy(:,2));
legend('Training Error');
xlabel('Iteration');
ylabel('Accuracy');
%}