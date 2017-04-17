function [ projection_vectors ] = PCA( fields,percent_field_retention )
    
    cov_matrix = cov(fields);
    [V,D] = eig(cov_matrix);
    
    % Get sum of all eigenvalues
    T = sum(D(:));
    
    count = size(D,1);
    while sum(sum(D(:,count:end)))/T < percent_field_retention
        count = count - 1;
    end
    projection_vectors = V(:,count:end);
end

