rng(0);
image_size = 40;
S = 9;
[~,classes,fields] = load_image_data(image_size,image_size);

deep_ann = Deep_ANN;

%% BASELINE
deep_ann = set_control_flow_vals(deep_ann,[0,0,1,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NNNN
deep_ann = set_control_flow_vals(deep_ann,[0,0,0,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YNNN
deep_ann = set_control_flow_vals(deep_ann,[1,0,0,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NYNN
deep_ann = set_control_flow_vals(deep_ann,[0,1,0,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NNNY
deep_ann = set_control_flow_vals(deep_ann,[0,0,0,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YYNN
deep_ann = set_control_flow_vals(deep_ann,[1,1,0,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YNYN
deep_ann = set_control_flow_vals(deep_ann,[1,0,1,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YNNY
deep_ann = set_control_flow_vals(deep_ann,[1,0,0,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NYYN
deep_ann = set_control_flow_vals(deep_ann,[0,1,1,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NYNY
deep_ann = set_control_flow_vals(deep_ann,[0,1,0,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NNYY
deep_ann = set_control_flow_vals(deep_ann,[0,0,1,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YYYN
deep_ann = set_control_flow_vals(deep_ann,[1,1,1,0]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YYNY
deep_ann = set_control_flow_vals(deep_ann,[1,1,0,1]);
rng(0);
[num_classes,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size);
rng(0);
[deep_ann,training_accuracy,validation_accuracy,testing_accuracy] = train_deep_ANN(deep_ann,training_fields,training_classes,validation_fields,validation_classes,testing_fields,testing_classes)
%{
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);
%}
%% YNYY
deep_ann = set_control_flow_vals(deep_ann,[1,0,1,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% NYYY
deep_ann = set_control_flow_vals(deep_ann,[0,1,1,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);

%% YYYY
deep_ann = set_control_flow_vals(deep_ann,[1,1,1,1]);
rng(0);
[~,s_training_accuracies,s_validation_accuracies,s_testing_accuracies] = cross_validate_deep_ANN(deep_ann,S,fields,classes);
mean_testing_accuracy = mean(s_testing_accuracies(:,2));
mean_training_accuracies = mean(s_training_accuracies(:,2:end));
mean_validation_accuracy = mean(s_validation_accuracies(:,2));
plot_training_accuracy(deep_ann,mean_testing_accuracy,[(1:1000)' mean_training_accuracies'],mean_validation_accuracy);
