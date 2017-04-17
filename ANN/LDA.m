function [ reduced_fields ] = LDA( fields, percent_field_retention )
    [num_classes, classes, fields] = load_image_data();
    new_num_features = ceil(size(fields, 2) * percent_field_retention);
    classified_fields = [classes, fields];
    class_means = nan(num_classes, length(fields));
    within_matrix = zeros(length(fields),length(fields));
    between_matrix = zeros(length(fields),length(fields));
    field_means = mean(fields);

    for i = 1:num_classes
        current_classes = classified_fields(classified_fields(:,1) == i,:);
        class_means(i,:) = mean(current_classes(:,2:end));
        class_pop = length(current_classes);
        within_matrix = within_matrix + (class_pop - 1) .* cov(current_classes(:,2:end));
        between_matrix = class_pop .* (class_means(i,:) - field_means)' * (class_means(i,:) - field_means);
    end

    [e_vectors, ~] = eigs(within_matrix \ between_matrix, new_num_features);
    reduced_fields = fields * e_vectors;
end

