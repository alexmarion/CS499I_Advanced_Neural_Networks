slope_sim_training = training_accuracies(:,2)\training_accuracies(:,3);
slope_sim_testing = testing_accuracies(:,2)\testing_accuracies(:,3);

fig = figure();
hold on;
plot(training_accuracies(:,1), training_accuracies(:,2));
plot(training_accuracies(:,1), training_accuracies(:,3));

plot(testing_accuracies(:,1), testing_accuracies(:,2));
plot(testing_accuracies(:,1), testing_accuracies(:,3));
legend('Training Accuracy Set 1','Training Accuracy Set 2','Testing Accuracy Set 1','Testing Accuracy Set 2','Location','southwest');
xlabel('Percent Field Retention');
ylabel('Accuracy');
title(sprintf('Training Slope Similarity Ratio: %f\nTesting Slope Similarity Ratio: %f',slope_sim_training,slope_sim_testing));
hold off;

saveas(fig,'../Latex/validation_imgs/field_retention_validation.png');
