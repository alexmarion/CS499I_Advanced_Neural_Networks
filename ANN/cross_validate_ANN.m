function [ s_training_and_testing_accuracies ] = cross_validate_ANN( S,num_classes,classes,fields )
    %rng(0);
    %[num_classes,classes,fields] = load_image_data(image_size,image_size);
    num_data_rows = size(fields,1);
    %S = 3;
    s_folds = cvpartition(num_data_rows,'k',S);

    shuffled_idxs = randperm(num_data_rows);
    shuffled_classes = classes(shuffled_idxs);
    shuffled_fields = fields(shuffled_idxs,:);

    num_hidden = 20;
    num_iters = 100;
    learning_rate = 0.5;
    field_retention = 0.95;

    s_training_accuracies = zeros(S,2);
    s_testing_accuracies = zeros(S,2);

    for i=1:S
        idxs = training(s_folds,i);

        training_idxs = find(idxs);
        s_training_fields = shuffled_fields(training_idxs,:);
        s_training_classes = shuffled_classes(training_idxs);

        testing_idxs = find(~idxs);
        s_testing_fields = shuffled_fields(testing_idxs,:);
        s_testing_classes = shuffled_classes(testing_idxs);

        [testing_accuracy,training_accuracy] = train_multi_class_ANN( ...
            num_classes,s_training_fields,s_training_classes,s_testing_fields,s_testing_classes, ... 
            num_hidden,                                 ...
            num_iters,                                  ...
            learning_rate,                              ...
            field_retention                             ...
        );
        s_training_accuracies(i,:) = [i,training_accuracy(end,2)];
        s_testing_accuracies(i,:) = [i,testing_accuracy];
    end

    figure();
    hold on;
    % plot(s_training_accuracies(:,1), s_training_accuracies(:,2),'b');
    % plot(s_testing_accuracies(:,1), s_testing_accuracies(:,2),'r');
    s_training_and_testing_accuracies = [s_training_accuracies(:,2) s_testing_accuracies(:,2)];
    bar(s_training_and_testing_accuracies);
    %bar(s_testing_accuracies(:,2),'r');
    legend('Training Accuracy','Testing Accuracy','Location','southwest')
    xlabel('Fold');
    ylabel('Accuracy');
    hold off;
end

