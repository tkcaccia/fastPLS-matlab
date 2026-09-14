function [U, D, V] = fastsvd(X, numComponents, options)
%FASTSVD Randomized truncated singular-value decomposition.
arguments
    X {mustBeNumeric,mustBeReal}
    numComponents (1,1) double {mustBeInteger,mustBePositive}
    options.Oversample (1,1) double {mustBeInteger,mustBeNonnegative} = 32
    options.Power (1,1) double {mustBeInteger,mustBeNonnegative} = 5
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 1
end
if ~isa(X, "single") && ~isa(X, "double"), X = double(X); end
[U, D, V] = fastpls_mex('fastsvd', X, numComponents, ...
    options.Oversample, options.Power, options.Seed);
end
