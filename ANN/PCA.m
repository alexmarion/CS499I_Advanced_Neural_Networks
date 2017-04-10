function [ reduced_fields ] = PCA( fields,percent_field_retention )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    num_features = size(fields, 2);
    new_num_features = ceil(num_features * percent_field_retention);
    
    cov_matrix = cov(fields);
    [V,~] = eigs(cov_matrix,new_num_features);
    reduced_fields = fields * V;
end

