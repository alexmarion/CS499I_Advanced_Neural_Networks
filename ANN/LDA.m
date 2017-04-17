[num_classes, classes, fields] = load_image_data();
classified_fields = [classes, fields];
class_means = [];

for i = 1:num_classes
    current_classes = classified_fields(classified_fields(:,1) == i,:);
    class_means = [class_means; mean(current_classes(:,2:end))];
end

within_matrix = zeros(length(fields),length(fields));

for i = 1:num_classes
    current_classes = classified_fields(classified_fields(:,1) == i,:);
    class_pop = length(current_classes);
    within_matrix = within_matrix + (class_pop - 1) .* cov(current_classes(:,2:end));
end

between_matrix = zeros(length(fields),length(fields));

field_means = mean(fields);

for i = 1:num_classes
    current_classes = classified_fields(classified_fields(:,1) == i,:);
    class_pop = length(current_classes);
    between_matrix = class_pop .* (class_means(i,:) - field_means)' .* (class_means(i,:) - field_means);
end

function [ separated_fields ] = run_LDA( fields )
    num_features = size(fields, 2);
    new_num_features = ceil(num_features * percent_field_retention);
end

