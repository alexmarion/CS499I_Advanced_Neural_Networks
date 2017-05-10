%% Load Data
rng(0);
data_selection_type = 0;
image_size = 40;
S = 9;
[~,classes,fields] = load_image_data(image_size,image_size);
[num_classes,training_fields,training_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size,data_selection_type);

% Perform PCA
percent_field_retention = 0.95;
projection_vectors = PCA(training_fields,percent_field_retention);
pca_training_fields = training_fields * projection_vectors;
pca_testing_fields = testing_fields * projection_vectors;

ann = ANN;
ann.eta = 1.5;
ann.should_plot = true;
ann.should_perform_PCA = false;
[~,~] = train_ANN(ann,pca_training_fields,training_classes,pca_testing_fields,testing_classes);

%% Iteration Testing
start_pt = 1;
end_pt = 2000;
training_iters = start_pt:10:end_pt;
training_accuracies = zeros(numel(training_iters),S + 1);
testing_accuracies = zeros(numel(training_iters),S + 1);

iter_test_ANN = ANN;

for i=1:numel(training_iters)
    num_iters = training_iters(i);
    disp(num_iters);
    iter_test_ANN.training_iters = num_iters;
%     [testing_accuracy,training_accuracy] = train_ANN(iter_test_ANN,training_fields,training_classes,testing_fields,testing_classes);
%     training_accuracies(i,:) = [num_iters,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [num_iters,testing_accuracy];

    rng(0);
    [s_training_accuracies,s_testing_accuracies] = cross_validate_ANN(iter_test_ANN,S,fields,classes);
    training_accuracies(i,:) = [num_iters,s_training_accuracies(:,2)'];
    testing_accuracies(i,:) = [num_iters,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Training Accuracy','Testing Accuracy','Location','southwest')
xlabel('Number of Iterations');
ylabel('Accuracy');
hold off;
% Save image and data
saveas(fig,'../Latex/figs/num_iterations_empirical.png');
save('../Data/num_iterations_empirical','training_accuracies','testing_accuracies')

%% Number of Hidden Nodes Testing
start_pt = 20;
end_pt = 1600;
num_hidden_nodes = start_pt:1:end_pt;
training_accuracies = zeros(numel(num_hidden_nodes),S + 1);
testing_accuracies = zeros(numel(num_hidden_nodes),S + 1);

hidden_nodes_ANN_test = ANN;
hidden_nodes_ANN_test.training_iters = 20;

for i=1:numel(num_hidden_nodes)
    num_hidden = num_hidden_nodes(i);
    disp(num_hidden);
    hidden_nodes_ANN_test.num_hidden_nodes = num_hidden;
%     [testing_accuracy,training_accuracy] = train_ANN(hidden_nodes_ANN_test,training_fields,training_classes,testing_fields,testing_classes);
%     training_accuracies(i,:) = [num_hidden,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [num_hidden,testing_accuracy];

    rng(0);
    [s_training_accuracies,s_testing_accuracies] = cross_validate_ANN(hidden_nodes_ANN_test,S,fields,classes);
    training_accuracies(i,:) = [num_hidden,s_training_accuracies(:,2)'];
    testing_accuracies(i,:) = [num_hidden,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Training Accuracy','Testing Accuracy','Location','southwest')
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
end_pt = 20;
learning_rates = start_pt:0.05:end_pt;
training_accuracies = zeros(numel(learning_rates),S + 1);
testing_accuracies = zeros(numel(learning_rates),S + 1);

eta_ANN_test = ANN;

for i=1:numel(learning_rates)
    learning_rate = learning_rates(i);
    disp(learning_rate);
    eta_ANN_test.eta = learning_rate;
%     [testing_accuracy,training_accuracy] = train_ANN(eta_ANN_test,training_fields,training_classes,testing_fields,testing_classes);
%     training_accuracies(i,:) = [learning_rate,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [learning_rate,testing_accuracy];

    rng(0);
    [s_training_accuracies,s_testing_accuracies] = cross_validate_ANN(eta_ANN_test,S,fields,classes);
    training_accuracies(i,:) = [learning_rate,s_training_accuracies(:,2)'];
    testing_accuracies(i,:) = [learning_rate,s_testing_accuracies(:,2)'];
end

fig = figure();
hold on;
plot(training_accuracies(:,1), mean(training_accuracies(:,2:S+1),2),'b');
plot(testing_accuracies(:,1), mean(testing_accuracies(:,2:S+1),2),'r');
legend('Training Accuracy','Testing Accuracy','Location','southwest')
xlabel('Learning Rate');
ylabel('Accuracy');
hold off;
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