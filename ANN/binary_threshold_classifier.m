function [ Accuracy, Precision, Recall ] = binary_threshold_classifier( threshold, output, testing_classes )
%binary_threshold_classifier.m Classifies binary data based on threshold

% If testing value is above threshold, set binary value 1
output(output > threshold) = 1;

% If testing value is below or equal to threshold, set binary value 0
output(output <= threshold) = 0;

% If the difference in the classes = 0 then they are the same
%num_correct = numel(find(~(output - testing_classes)));

TP = size(testing_classes(testing_classes(output == 1) == 1), 1);
FP = size(testing_classes(testing_classes(output == 1) == 0), 1);
TN = size(testing_classes(testing_classes(output == 0) == 0), 1);
FN = size(testing_classes(testing_classes(output == 0) == 1), 1);

Precision = TP/(TP + FP);
Recall = TP/(TP + FN);
Accuracy = (TP + TN)/(TP + TN + FP + FN);
end

