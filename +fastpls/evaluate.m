function result = evaluate(observed, predicted, options)
%EVALUATE Calculate classification or regression prediction metrics.
arguments
    observed
    predicted
    options.YTrain = []
    options.ByColumn (1, 1) logical = true
    options.RelativeEpsilon (1, 1) double = eps
end
isLabels = (iscategorical(observed) || isstring(observed) || ...
    iscellstr(observed) || islogical(observed)) && isvector(observed);
if isLabels
    observed = string(observed(:));
    predicted = string(predicted);
    if isvector(predicted), predicted = predicted(:); end
    if size(predicted, 1) ~= numel(observed)
        error("fastPLS:DimensionMismatch", ...
            "Observed and predicted labels have different lengths.");
    end
    labels = unique([observed; predicted(:, 1)], "stable");
    confusion = zeros(numel(labels), numel(labels));
    recall = zeros(numel(labels), 1);
    precision = zeros(numel(labels), 1);
    f1 = zeros(numel(labels), 1);
    for index = 1:numel(labels)
        truth = observed == labels(index);
        estimate = predicted(:, 1) == labels(index);
        truePositive = sum(truth & estimate);
        recall(index) = truePositive / max(1, sum(truth));
        precision(index) = truePositive / max(1, sum(estimate));
        if precision(index) + recall(index) > 0
            f1(index) = 2 * precision(index) * recall(index) / ...
                (precision(index) + recall(index));
        end
        for predictedIndex = 1:numel(labels)
            confusion(index, predictedIndex) = sum( ...
                truth & predicted(:, 1) == labels(predictedIndex));
        end
    end
    accuracy = mean(predicted(:, 1) == observed);
    counts = sum(confusion, 2);
    noInformation = max(counts) / numel(observed);
    chance = (sum(confusion, 2)' * sum(confusion, 1)') / numel(observed)^2;
    if chance < 1, kappa = (accuracy - chance) / (1 - chance); else, kappa = NaN; end
    metrics = struct( ...
        "Accuracy", accuracy, ...
        "NoInformationRate", noInformation, ...
        "LiftAccuracy", accuracy / noInformation, ...
        "BalancedAccuracy", mean(recall), ...
        "MacroPrecision", mean(precision), ...
        "MacroRecall", mean(recall), ...
        "MacroF1", mean(f1), ...
        "Kappa", kappa, ...
        "TopAccuracy", mean(any(predicted == observed, 2)), ...
        "Top", size(predicted, 2));
    result = metrics;
    result.Task = "classification";
    result.Metrics = metrics;
    result.Classes = labels;
    result.Confusion = confusion;
    result.PerClass = table(labels, precision, recall, f1, ...
        VariableNames=["Class", "Precision", "Recall", "F1"]);
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
rmsd = sqrt(mean(residual .^ 2, "all"));
q2 = NaN;
if ~isempty(options.YTrain)
    training = double(options.YTrain);
    if size(training, 2) ~= size(observed, 2)
        error("fastPLS:DimensionMismatch", ...
            "YTrain and observed must have the same response columns.");
    end
    q2Total = sum((observed - mean(training, 1)) .^ 2, "all");
    if q2Total > 0, q2 = 1 - press / q2Total; end
end
selected = abs(observed) > options.RelativeEpsilon;
relative = abs(residual(selected) ./ observed(selected));
observedVector = observed(:);
predictedVector = predicted(:);
standardDeviation = sqrt(sum(centered .^ 2, "all") / ...
    max(numel(observed) - 1, 1));
pearson = localCorrelation(observedVector, predictedVector);
spearman = localCorrelation(averageRanks(observedVector), averageRanks(predictedVector));
metrics = struct( ...
    "R2", conditional(total > 0, 1 - press / total, NaN), ...
    "Q2", q2, ...
    "RMSD", rmsd, ...
    "RMSE", rmsd, ...
    "MAE", mean(abs(residual), "all"), ...
    "Bias", mean(predicted - observed, "all"), ...
    "MedianRelativeErrorPercent", conditional(~isempty(relative), 100 * median(relative), NaN), ...
    "MAPEPercent", conditional(~isempty(relative), 100 * mean(relative), NaN), ...
    "RPD", standardDeviation / rmsd, ...
    "PearsonR", pearson, ...
    "SpearmanR", spearman);
result = metrics;
result.Task = "regression";
result.Metrics = metrics;
if options.ByColumn && size(observed, 2) > 1
    for column = 1:size(observed, 2)
        trainingColumn = [];
        if ~isempty(options.YTrain), trainingColumn = options.YTrain(:, column); end
        current = fastpls.evaluate(observed(:, column), predicted(:, column), ...
            YTrain=trainingColumn, ByColumn=false);
        if column == 1
            perResponse = repmat(current.Metrics, 1, size(observed, 2));
        else
            perResponse(column) = current.Metrics;
        end
    end
    result.PerResponse = perResponse;
end
end

function value = conditional(test, yes, no)
if test, value = yes; else, value = no; end
end

function value = localCorrelation(left, right)
left = left - mean(left);
right = right - mean(right);
denominator = norm(left) * norm(right);
if denominator > 0, value = (left' * right) / denominator; else, value = NaN; end
end

function ranks = averageRanks(values)
[sorted, order] = sort(values);
ranks = zeros(size(values));
first = 1;
while first <= numel(values)
    last = first;
    while last < numel(values) && sorted(last + 1) == sorted(first)
        last = last + 1;
    end
    ranks(order(first:last)) = (first + last) / 2;
    first = last + 1;
end
end
