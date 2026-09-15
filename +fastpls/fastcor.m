function value = fastcor(A, B, options)
%FASTCOR Pearson correlation between matrix rows or columns.
arguments
    A {mustBeNumeric,mustBeReal}
    B {mustBeNumeric,mustBeReal} = []
    options.ByRow (1, 1) logical = true
    options.Diagonal (1, 1) logical = true
end
if isempty(B), B = A; supplied = false; else, supplied = true; end
if options.ByRow
    if size(A, 2) ~= size(B, 2)
        error("fastPLS:DimensionMismatch", ...
            "A and B must have the same number of columns.");
    end
    A = A - mean(A, 2);
    B = B - mean(B, 2);
    scaleA = sqrt(sum(double(A) .^ 2, 2));
    scaleB = sqrt(sum(double(B) .^ 2, 2));
    value = (A * B') ./ (scaleA * scaleB');
else
    if size(A, 1) ~= size(B, 1)
        error("fastPLS:DimensionMismatch", ...
            "A and B must have the same number of rows.");
    end
    A = A - mean(A, 1);
    B = B - mean(B, 1);
    scaleA = sqrt(sum(double(A) .^ 2, 1));
    scaleB = sqrt(sum(double(B) .^ 2, 1));
    value = (A' * B) ./ (scaleA' * scaleB);
end
if any(scaleA == 0) || any(scaleB == 0)
    error("fastPLS:ConstantInput", ...
        "Correlation is undefined for constant rows or columns.");
end
if supplied && options.Diagonal
    if size(value, 1) ~= size(value, 2)
        error("fastPLS:DimensionMismatch", ...
            "Diagonal=true requires matching rows or columns.");
    end
    value = diag(value);
end
end
