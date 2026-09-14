function value = fastcor(X)
%FASTCOR Pearson correlation between matrix columns.
arguments
    X {mustBeNumeric,mustBeReal}
end
X = X - mean(X, 1);
norms = sqrt(sum(double(X) .^ 2, 1));
if any(norms == 0), error("fastPLS:ConstantColumn", "Correlation is undefined for constant columns."); end
value = (X' * X) ./ (norms' * norms);
end
