function [ std_fields, m, s ] = standardize_data( fields )
%standardize_data standardizes fields passed in an NxM matrix
    m = mean(fields);
    s = std(fields);

    std_fields = fields - repmat(m,size(fields,1),1);
    std_fields = std_fields ./ repmat(s,size(fields,1),1);
end

