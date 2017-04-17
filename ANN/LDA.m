function [ reduced_fields ] = LDA(classes, fields, percent_field_retention)
    num_features = length(fields);
    num_classes = length(unique(classes));
    new_num_features = ceil(num_features * percent_field_retention);
    classified_fields = [classes, fields];
    class_means = nan(num_classes, num_features);
    within_matrix = zeros(num_features,num_features);
    between_matrix = zeros(num_features,num_features);
    field_means = mean(fields);

    for i = 1:num_classes
        current_classes = classified_fields(classified_fields(:,1) == i,:);
        class_means(i,:) = mean(current_classes(:,2:end));
        class_pop = length(current_classes);
        within_matrix = within_matrix + (class_pop - 1) .* cov(current_classes(:,2:end));
        between_matrix = between_matrix + class_pop .* (class_means(i,:) - field_means)' * (class_means(i,:) - field_means);
    end

    [e_vectors, ~] = eigs(within_matrix \ between_matrix, 2);
    reduced_fields = fields * e_vectors;
end

