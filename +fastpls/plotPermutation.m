function axesHandle = plotPermutation(result, options)
%PLOTPERMUTATION Plot permuted and observed selection statistics.
arguments
    result (1, 1) struct
    options.Parent = []
end
required = ["PermutationSampled", "PermutationObserved", "PermutationMetric"];
if ~all(isfield(result, required))
    error("fastPLS:InvalidPermutation", ...
        "The input does not contain a fastPLS permutation result.");
end
if isempty(options.Parent)
    axesHandle = axes(figure);
else
    axesHandle = options.Parent;
end
sampled = result.PermutationSampled(:);
scatter(axesHandle, (1:numel(sampled))', sampled, 24, "filled", ...
    DisplayName="Permuted");
hold(axesHandle, "on");
yline(axesHandle, result.PermutationObserved, "--", ...
    DisplayName="Observed");
hold(axesHandle, "off");
xlabel(axesHandle, "Permutation");
ylabel(axesHandle, string(result.PermutationMetric));
legend(axesHandle, Location="best");
end
