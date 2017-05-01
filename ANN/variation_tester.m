rng(0);
data_selection_type = 0;
image_size = 40;
[num_classes,training_fields,training_classes,testing_fields,testing_classes] = load_and_shuffle_data(image_size,data_selection_type);

ann = ANN;

% BASELINE
ann = set_control_flow_vals(ann,[0,0,1,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NNNN
ann = set_control_flow_vals(ann,[0,0,0,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YNNN
ann = set_control_flow_vals(ann,[1,0,0,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NYNN
ann = set_control_flow_vals(ann,[0,1,0,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NNNY
ann = set_control_flow_vals(ann,[0,0,0,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YYNN
ann = set_control_flow_vals(ann,[1,1,0,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YNYN
ann = set_control_flow_vals(ann,[1,0,1,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YNNY
ann = set_control_flow_vals(ann,[1,0,0,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NYYN
ann = set_control_flow_vals(ann,[0,1,1,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NYNY
ann = set_control_flow_vals(ann,[0,1,0,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NNYY
ann = set_control_flow_vals(ann,[0,0,1,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YYYN
ann = set_control_flow_vals(ann,[1,1,1,0]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YYNY
ann = set_control_flow_vals(ann,[1,1,0,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YNYY
ann = set_control_flow_vals(ann,[1,0,1,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% NYYY
ann = set_control_flow_vals(ann,[0,1,1,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);

% YYYY
ann = set_control_flow_vals(ann,[1,1,1,1]);
[testing_accuracy,training_accuracy] = train_ANN(ann,training_fields,training_classes,testing_fields,testing_classes);
plot_training_accuracy(ann,testing_accuracy,training_accuracy);
