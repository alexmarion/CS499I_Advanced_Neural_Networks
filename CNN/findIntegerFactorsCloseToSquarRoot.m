function [a, b] =  findIntegerFactorsCloseToSquarRoot(n)
    amax = floor(sqrt(n));
    if 0 == rem(n, amax)
        a = amax;
        b = n / a;
        return;
    end
    primeFactors  = factor(n);
    candidates = [1];
    for i=1:numel(primeFactors)
        f = primeFactors(i);
        candidates  = union(candidates, f .* candidates);
        candidates(candidates > amax) = [];
    end
    a = candidates(end);
    b = n / a;
end