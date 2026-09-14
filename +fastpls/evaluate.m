function metrics = evaluate(observed, predicted)
%EVALUATE Calculate classification or regression prediction metrics.
if isvector(observed) && (isvector(predicted) || size(predicted, 2) > 1) && ...
        (~isnumeric(observed) || islogical(observed) || iscategorical(observed))
    observed = observed(:);
    if isvector(predicted)
        predicted = predicted(:);
    end
    if size(predicted, 1) ~= numel(observed)
        error("fastPLS:DimensionMismatch", ...
            "Observed and predicted labels have different lengths.");
    end
    correct = predicted == observed;
    labels = unique(observed);
    recall = zeros(numel(labels), 1);
    for index = 1:numel(labels)
        selected = observed == labels(index);
        recall(index) = mean(predicted(selected, 1) == labels(index));
    end
    metrics = struct( ...
        "Accuracy", mean(correct(:, 1)), ...
        "BalancedAccuracy", mean(recall), ...
        "TopAccuracy", mean(any(correct, 2)), ...
        "Top", size(predicted, 2));
    return
end
observed = double(observed);
predicted = double(predicted);
if ~isequal(size(observed), size(predicted))
    error("fastPLS:DimensionMismatch", ...
        "Observed and predicted responses have different dimensions.");
end
residual = observed - predicted;
press = sum(residual .^ 2, "all");
centered = observed - mean(observed, 1);
total = sum(centered .^ 2, "all");
metrics = struct( ...
    "R2", 1 - press / total, ...
    "RMSD", sqrt(mean(residual .^ 2, "all")), ...
    "MAE", mean(abs(residual), "all"));
end
