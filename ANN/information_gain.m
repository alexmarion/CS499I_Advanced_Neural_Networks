% function heatmap = information_gain(num_classes, classes, fields)

[num_classes, classes, fields] = load_image_data(40,40);
ig = zeros(1,1600);

ent = 0;
ent_bar = 0;
for feature_value = 1:num_classes
   % Grab probability that uh... hold on
   ent = ent - (1/num_classes * log(1/num_classes)/log(15));            
end

for feature_num = 1:length(fields)
    for feature_value = 1:255
        prob_val = numel(fields(fields(:,feature_num) == feature_value), feature_num) / size(fields,1);
        sum_prob_class = 0;
        for class_num = 1:num_classes
            prob_class = 1/num_classes;
            class_fields = fields(classes == class_num, feature_num);
            prob_class_k = numel(fields(class_fields == feature_value)) / numel(class_fields);
            c_given_v = ((prob_class * prob_class_k) / prob_val);
            
            if c_given_v == 0
                continue
            end
            
            sum_prob_class = sum_prob_class - c_given_v * (log(c_given_v)/log(15));
        end
        ent_bar = ent_bar + prob_val * sum_prob_class;
    end
    ig(feature_num) = ent + ent_bar;
    ent_bar = 0;
end
