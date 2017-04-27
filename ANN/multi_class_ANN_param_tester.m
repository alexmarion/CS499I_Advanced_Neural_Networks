%% Load Data
rng(0);
data_selection_type = 0;
image_size = 40;
[num_classes,training_fields,training_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size,data_selection_type);

%% Iteration Testing

start_pt = 1;
end_pt = 2000;
training_iters = start_pt:10:end_pt;
training_accuracies = zeros(numel(training_iters),2);
testing_accuracies = zeros(numel(training_iters),2);

num_hidden = 20;
learning_rate = 0.5;
field_retention = 0.95;

for i=1:numel(training_iters)
    num_iters = training_iters(i);
    disp(num_iters);
    [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
        num_classes,training_fields,training_classes,testing_fields,testing_classes, ... 
        num_hidden,                                 ...
        num_iters,                                  ...
        learning_rate,                              ...
        field_retention                             ...
    );
    training_accuracies(i,:) = [num_iters,training_accuracy(end,2)];
    testing_accuracies(i,:) = [num_iters,testing_accuracy];
end

figure();
hold on;
fit_deg = 2;
plot(training_accuracies(:,1), training_accuracies(:,2),'b');
[xPts,yPts] = get_trendline(training_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'r');
[xPts,yPts] = get_trendline(testing_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Number of Iterations');
ylabel('Accuracy');
hold off;

%% Number of Hidden Nodes Testing
%{
start_pt = 20;
end_pt = 1600;
num_hidden_nodes = start_pt:20:end_pt;
training_accuracies = zeros(numel(num_hidden_nodes),2);
testing_accuracies = zeros(numel(num_hidden_nodes),2);

num_iters = 50;
learning_rate = 0.5;
field_retention = 0.95;

for i=1:numel(num_hidden_nodes)
    num_hidden = num_hidden_nodes(i);
    disp(num_hidden);
    [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
        num_classes,training_fields,training_classes,testing_fields,testing_classes, ... 
        num_hidden,                                 ...
        num_iters,                                  ...
        learning_rate,                              ...
        field_retention                             ...
    );
    training_accuracies(i,:) = [num_hidden,training_accuracy(end,2)];
    testing_accuracies(i,:) = [num_hidden,testing_accuracy];
end

figure();
hold on;
fit_deg = 2;
plot(training_accuracies(:,1), training_accuracies(:,2),'bx');
[xPts,yPts] = get_trendline(training_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'or');
[xPts,yPts] = get_trendline(testing_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Number of Hidden Nodes');
ylabel('Accuracy');
hold off;
%}

%% Size of Image Testing
%{
start_pt = 10;
end_pt = 50;
image_sizes = start_pt:1:end_pt;
training_accuracies = zeros(numel(image_sizes),2);
testing_accuracies = zeros(numel(image_sizes),2);

num_iters = 1000;
num_hidden = 20;
learning_rate = 0.5;
field_retention = 0.95;

for i=1:numel(image_sizes)
    image_size = image_sizes(i);
    disp(image_size);
    
    % Must re-seed every time
    rng(0);
    [num_classes,training_fields,training_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size,data_selection_type);

    [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
        num_classes,training_fields,training_classes,testing_fields,testing_classes, ... 
        num_hidden,                                 ...
        num_iters,                                  ...
        learning_rate,                              ...
        field_retention                             ...
    );
    training_accuracies(i,:) = [image_size,training_accuracy(end,2)];
    testing_accuracies(i,:) = [image_size,testing_accuracy];
end

figure();
hold on;
fit_deg = 2;
plot(training_accuracies(:,1), training_accuracies(:,2),'bx');
[xPts,yPts] = get_trendline(training_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'or');
[xPts,yPts] = get_trendline(testing_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Image Size');
ylabel('Accuracy');
hold off;
%}

%% Learning Rate Testing
%{
start_pt = 0.05;
end_pt = 20;
learning_rates = start_pt:0.05:end_pt;
training_accuracies = zeros(numel(learning_rates),2);
testing_accuracies = zeros(numel(learning_rates),2);

num_iters = 1000;
num_hidden = 20;
field_retention = 0.95;


for i=1:numel(learning_rates)
    learning_rate = learning_rates(i);
    disp(learning_rate);
    [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
        num_classes,training_fields,training_classes,testing_fields,testing_classes, ... 
        num_hidden,                                 ...
        num_iters,                                  ...
        learning_rate,                              ...
        field_retention                             ...
    );
    training_accuracies(i,:) = [learning_rate,training_accuracy(end,2)];
    testing_accuracies(i,:) = [learning_rate,testing_accuracy];
end


figure();
hold on;
fit_deg = 2;
plot(training_accuracies(:,1), training_accuracies(:,2),'bx');
[xPts,yPts] = get_trendline(training_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'or');
[xPts,yPts] = get_trendline(testing_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Learning Rate');
ylabel('Accuracy');
hold off;
%}

%% Percent Field Retention
%{
start_pt = 0.01;
end_pt = 1;
percent_field_retention = start_pt:0.01:end_pt;
training_accuracies = zeros(numel(percent_field_retention),2);
testing_accuracies = zeros(numel(percent_field_retention),2);

num_iters = 1000;
num_hidden = 20;
learning_rate = 0.95;

for i=1:numel(percent_field_retention)
    percent_retention = percent_field_retention(i);
    disp(percent_retention); 
    [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
        num_classes,training_fields,training_classes,testing_fields,testing_classes, ... 
        num_hidden,                                 ...
        num_iters,                                  ...
        learning_rate,                              ...
        percent_retention                           ...
    );
    training_accuracies(i,:) = [percent_retention,training_accuracy(end,2)];
    testing_accuracies(i,:) = [percent_retention,testing_accuracy];
end

figure();
hold on;
fit_deg = 2;
plot(training_accuracies(:,1), training_accuracies(:,2),'xb');
[xPts,yPts] = get_trendline(training_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'or');
[xPts,yPts] = get_trendline(testing_accuracies,start_pt,end_pt,fit_deg);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Percent Field Retention');
ylabel('Accuracy');
hold off;
%}

%% Best Value Test
% testing_accuracies(testing_accuracies(:,2)==max(testing_accuracies(:,2)))
%{
img_size = 30;
learning_rate = 0.75;
num_hidden = 1200;
iterations = 1000;
retention = 0.97;

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