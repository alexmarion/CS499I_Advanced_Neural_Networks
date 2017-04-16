%% Accuracy Testing
training_iters = 0:10:10000;
accuracies = zeros(numel(training_iters),2);

for i=1:numel(training_iters)
    num_iters = training_iters(i);
    [accuracy,training_error] = train_multi_class_ANN(true,false,true,false,false,20,num_iters);
    accuracies(i,:) = [num_iters,accuracy];
end

figure();
plot(accuracies(:,1), accuracies(:,2));
legend('Training Accuracy');
xlabel('Iterations');
ylabel('Accuracy');

%% Number of Hidden Nodes Testing
% num_hidden_nodes = 20:20:1600;
% accuracies = zeros(numel(num_hidden_nodes),2);
% for i=1:numel(num_hidden_nodes)
%     num_hidden = num_hidden_nodes(i);
%     disp(num_hidden);
%     [accuracy,training_error] = train_multi_class_ANN(true,false,true,false,false,num_hidden,1000);
%     accuracies(i,:) = [num_hidden,accuracy];
% end
% 
% figure();
% plot(accuracies(:,1), accuracies(:,2));
% legend('Training Accuracy');
% xlabel('Number of Hidden Nodes');
% ylabel('Accuracy');
