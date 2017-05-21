rng(0);
image_size = 40;
S = 9;
%[num_classes,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size);
[~,classes,fields] = load_image_data(image_size,image_size);

ann = ANN;

% BASELINE
ann = set_control_flow_vals(ann,[0,0,1,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NNNN
ann = set_control_flow_vals(ann,[0,0,0,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YNNN
ann = set_control_flow_vals(ann,[1,0,0,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NYNN
ann = set_control_flow_vals(ann,[0,1,0,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NNNY
ann = set_control_flow_vals(ann,[0,0,0,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YYNN
ann = set_control_flow_vals(ann,[1,1,0,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YNYN
ann = set_control_flow_vals(ann,[1,0,1,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YNNY
ann = set_control_flow_vals(ann,[1,0,0,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NYYN
ann = set_control_flow_vals(ann,[0,1,1,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NYNY
ann = set_control_flow_vals(ann,[0,1,0,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NNYY
ann = set_control_flow_vals(ann,[0,0,1,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YYYN
ann = set_control_flow_vals(ann,[1,1,1,0]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YYNY
ann = set_control_flow_vals(ann,[1,1,0,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YNYY
ann = set_control_flow_vals(ann,[1,0,1,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% NYYY
ann = set_control_flow_vals(ann,[0,1,1,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

% YYYY
ann = set_control_flow_vals(ann,[1,1,1,1]);
rng(0);
[ann,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_ANN(ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);
