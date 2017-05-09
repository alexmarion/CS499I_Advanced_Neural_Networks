S = size(training_accuracies,2) - 1;

slope_sim_training = training_accuracies(:,2)\training_accuracies(:,3);
slope_sim_testing = testing_accuracies(:,2)\testing_accuracies(:,3);

fig = figure();
hold on;
for s=2:S+1
    plot(training_accuracies(:,1),training_accuracies(:,s));
    plot(testing_accuracies(:,1), testing_accuracies(:,s));
end

std_training = std(mean(testing_accuracies(:,2:S+1)));
std_testing = std(mean(training_accuracies(:,2:S+1)));

%legend('Training Accuracy Set 1','Training Accuracy Set 2','Testing Accuracy Set 1','Testing Accuracy Set 2','Location','southwest');
xlabel('Percent Field Retention');
ylabel('Accuracy');
title(sprintf('Training Accuracy Standard Deviation: %f\nTesting Accuracy Standard Deviation: %f',std_training,std_testing));
%title(sprintf('Training Slope Similarity Ratio: %f\nTesting Slope Similarity Ratio: %f',slope_sim_training,slope_sim_testing));
hold off;

saveas(fig,'../Latex/validation_imgs/field_retention_validation.png');
