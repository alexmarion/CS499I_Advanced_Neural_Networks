%% Accuracy Testing
rng(0);

% training_iters = 100:100:10000;
% training_accuracies = zeros(numel(training_iters),2);
% testing_accuracies = zeros(numel(training_iters),2);
% 
% for i=1:numel(training_iters)
%     num_iters = training_iters(i);
%     disp(num_iters);
%     [testing_accuracy,training_accuracy] = train_multi_class_ANN(true,false,true,true,20,num_iters,40);
%     training_accuracies(i,:) = [num_iters,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [num_iters,testing_accuracy];
% end
% 
% figure();
% hold on;
% plot(training_accuracies(:,1), training_accuracies(:,2));
% plot(testing_accuracies(:,1), testing_accuracies(:,2));
% legend('Training Accuracy','Testing Accuracy');
% xlabel('Number of Iterations');
% ylabel('Accuracy');
% hold off;

%% Number of Hidden Nodes Testing
% num_hidden_nodes = 20:20:1600;
% training_accuracies = zeros(numel(num_hidden_nodes),2);
% testing_accuracies = zeros(numel(num_hidden_nodes),2);
% 
% for i=1:numel(num_hidden_nodes)
%     num_hidden = num_hidden_nodes(i);
%     disp(num_hidden);
%     [testing_accuracy,training_accuracy] = train_multi_class_ANN(true,false,true,true,num_hidden,10,40);
%     training_accuracies(i,:) = [num_hidden,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [num_hidden,testing_accuracy];
% end
% 
% figure();
% hold on;
% plot(training_accuracies(:,1), training_accuracies(:,2));
% plot(testing_accuracies(:,1), testing_accuracies(:,2));
% legend('Training Accuracy','Testing Accuracy');
% xlabel('Number of Hidden Nodes');
% ylabel('Accuracy');
% hold off;

% %% Size of Image Testing
% image_sizes = 10:5:200;
% training_accuracies = zeros(numel(image_sizes),2);
% testing_accuracies = zeros(numel(image_sizes),2);
% 
% tic
% 
% for(i=1:numel(image_sizes))
%     break;
%     image_size = image_sizes(i);
%     disp(image_size);
%     [testing_accuracy,training_accuracy] = train_multi_class_ANN(true,false,true,true,20,500,image_size);
%     training_accuracies(i,:) = [image_size,training_accuracy(end,2)];
%     testing_accuracies(i,:) = [image_size,testing_accuracy];
%     break;
% end
% 
% toc
% 
% figure();
% hold on;
% plot(training_accuracies(:,1), training_accuracies(:,2));
% plot(testing_accuracies(:,1), testing_accuracies(:,2));
% legend('Training Accuracy','Testing Accuracy');
% xlabel('Image Size');
% ylabel('Accuracy');
% hold off;

%% Size of Image Testing
learning_rates = 0.05:0.05:20;
training_accuracies = zeros(numel(learning_rates),2);
testing_accuracies = zeros(numel(learning_rates),2);

tic

for i=1:numel(learning_rates)
    learning_rate = learning_rates(i);
    disp(learning_rate);
    [testing_accuracy,training_accuracy] = train_multi_class_ANN(true,false,true,true,20,50,30,learning_rate);
    training_accuracies(i,:) = [learning_rate,training_accuracy(end,2)];
    testing_accuracies(i,:) = [learning_rate,testing_accuracy];
end

toc

figure();
hold on;
plot(training_accuracies(:,1), training_accuracies(:,2),'bx');
[xPts,yPts] = get_trendline(training_accuracies,10,100,3);
plot(xPts,yPts,'b');

plot(testing_accuracies(:,1), testing_accuracies(:,2),'or');
[xPts,yPts] = get_trendline(testing_accuracies,10,100,3);
plot(xPts,yPts,'r');

legend('Training Accuracy','Training Trendline','Testing Accuracy','Testing Trendline');
xlabel('Image Size');
ylabel('Accuracy');
hold off;