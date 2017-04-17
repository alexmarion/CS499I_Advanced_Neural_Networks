%% Accuracy Testing
rng(0);
% training_iters = 0:100:10000;
% accuracies = zeros(numel(training_iters),2);
% 
% for i=1:numel(training_iters)
%     num_iters = training_iters(i);
%     disp(num_iters);
%     [accuracy,training_error] = train_multi_class_ANN(true,false,true,false,false,20,num_iters,40);
%     accuracies(i,:) = [num_iters,accuracy];
% end
% 
% figure();
% plot(accuracies(:,1), accuracies(:,2));
% legend('Training Accuracy');
% xlabel('Iterations');
% ylabel('Accuracy');

%% Number of Hidden Nodes Testing
% num_hidden_nodes = 20:20:1600;
% accuracies = zeros(numel(num_hidden_nodes),2);
% for i=1:numel(num_hidden_nodes)
%     num_hidden = num_hidden_nodes(i);
%     disp(num_hidden);
%     [accuracy,training_error] = train_multi_class_ANN(true,false,true,false,false,num_hidden,1000,40);
%     accuracies(i,:) = [num_hidden,accuracy];
% end
% 
% figure();
% plot(accuracies(:,1), accuracies(:,2));
% legend('Training Accuracy');
% xlabel('Number of Hidden Nodes');
% ylabel('Accuracy');

%% Size of Image Testing
image_sizes = 1:300;
accuracies = zeros(numel(image_sizes),2);
for i=1:numel(image_sizes)
    image_size = image_sizes(i);
    disp(image_size);
    [accuracy,training_error] = train_multi_class_ANN(true,false,true,false,false,20,1000,image_size);
    accuracies(i,:) = [image_size,accuracy];
end

figure();
plot(accuracies(:,1), accuracies(:,2));
legend('Training Accuracy');
xlabel('Image Size');
ylabel('Accuracy');
