function result = plsDoubleCV(Xdata, Ydata, options)
%PLSDOUBLECV Repeated nested cross-validation with optional permutation test.
arguments
    Xdata {mustBeNumeric,mustBeReal}
    Ydata
    options.NumComponents = 2
    options.Constrain = []
    options.Runs (1, 1) double = 1
    options.InnerFolds = 10
    options.OuterFolds = 10
    options.PermutationTest (1, 1) logical = false
    options.Times (1, 1) double = 100
    options.Seed (1, 1) double = 1
    options.Scaling = "centering"
    options.Method = "simpls"
    options.Backend = "cpu"
    options.OrthogonalComponents = 1
    options.Kernel = "linear"
    options.Gamma = []
    options.Degree = 3
    options.Offset = 1
    options.Classifier = ""
    options.Selection = "auto"
    options.Oversample = 32
    options.Power = 5
    options.ByColumn (1, 1) logical = false
end
validatePositiveInteger(options.Runs, "Runs");
if options.PermutationTest, validatePositiveInteger(options.Times, "Times"); end
if lower(string(options.Backend)) ~= "cpu"
    error("fastPLS:UnavailableBackend", ...
        "fastPLS-matlab currently provides only the CPU backend.");
end
Xdata = numericMatrix(Xdata, "Xdata");
classification = isClassification(Ydata, options.Classifier);
if classification, Ydata = string(Ydata(:)); else, Ydata = numericMatrix(Ydata, "Ydata"); end
if isempty(options.Constrain), groups = (1:size(Xdata, 1))'; else, groups = options.Constrain(:); end
if numel(groups) ~= size(Xdata, 1)
    error("fastPLS:InvalidGroups", "Constrain must contain one group per sample.");
end
plans = cell(options.Runs, 1);
for run = 1:options.Runs
    outer = makeFolds(size(Xdata, 1), options.OuterFolds, groups, ...
        conditional(classification, Ydata, []), options.Seed + 10000 * (run - 1));
    inner = cell(max(outer), 1);
    for fold = unique(outer(:))'
        train = outer ~= fold;
        inner{fold} = makeFolds(sum(train), options.InnerFolds, groups(train), ...
            conditional(classification, Ydata(train), []), ...
            options.Seed + 10000 * (run - 1) + fold);
    end
    plans{run} = struct("Outer", outer, "Inner", {inner});
end
[runs, combined] = runNested(Xdata, Ydata, groups, plans, options, classification);
selected = [runs.BestNumComponents];
result = struct( ...
    "Results", runs, ...
    "Prediction", combined, ...
    "Metrics", fastpls.evaluate(Ydata, combined, ByColumn=options.ByColumn), ...
    "BestNumComponents", mode(selected), ...
    "SelectionMetric", runs(1).SelectionMetric, ...
    "Q2Y", [runs.Q2Y], ...
    "R2Y", [runs.R2Y], ...
    "Method", string(options.Method), ...
    "Backend", string(options.Backend));
if options.PermutationTest
    stream = RandStream("mt19937ar", Seed=options.Seed + 900000);
    sampled = nan(options.Times, 1);
    errors = strings(options.Times, 1);
    completed = 0;
    for iteration = 1:options.Times
        index = permutationIndices(groups, stream);
        try
            nullRuns = runNested(Xdata, Ydata(index, :), groups, plans, options, classification);
            completed = completed + 1;
            sampled(completed) = median([nullRuns.MetricValue]);
        catch exception
            errors(iteration) = string(exception.message);
        end
    end
    sampled = sampled(1:completed);
    errors(errors == "") = [];
    observed = median([runs.MetricValue]);
    minimize = ismember(runs(1).SelectionMetric, ["rmsd", "mae", "mape_percent"]);
    if minimize, extreme = sum(sampled <= observed); else, extreme = sum(sampled >= observed); end
    complete = completed == options.Times && isempty(errors);
    result.PermutationMetric = runs(1).SelectionMetric;
    result.PermutationObserved = observed;
    result.PermutationSampled = sampled;
    if complete
        result.PValue = (extreme + 1) / (completed + 1);
    else
        result.PValue = NaN;
    end
    result.PermutationValid = complete;
    result.PermutationRequested = options.Times;
    result.PermutationCompleted = completed;
    result.PermutationFailed = options.Times - completed;
    result.PermutationErrors = errors;
    result.PermutationUnit = conditional( ...
        isempty(options.Constrain), "rows", "exchangeability blocks");
end

function validatePositiveInteger(value, name)
if ~isscalar(value) || ~isfinite(value) || value < 1 || fix(value) ~= value
    error("fastPLS:InvalidControl", "%s must be a positive integer.", name);
end
end
end

function [runs, combined] = runNested(X, Y, groups, plans, options, classification)
predictions = cell(options.Runs, 1);
for run = 1:options.Runs
    outer = plans{run}.Outer;
    if classification, prediction = strings(size(Y)); else, prediction = zeros(size(Y), "like", Y); end
    selected = zeros(max(outer), 1);
    q2Press = 0;
    q2Reference = 0;
    trainingR2 = nan(max(outer), 1);
    for fold = unique(outer(:))'
        train = outer ~= fold;
        test = ~train;
        inner = fastpls.plsSingleCV( ...
            X(train, :), Y(train, :), ...
            NumComponents=options.NumComponents, Constrain=groups(train), ...
            Scaling=options.Scaling, Method=options.Method, Backend=options.Backend, ...
            Seed=options.Seed + 10000 * (run - 1) + fold, ...
            KFold=options.InnerFolds, OrthogonalComponents=options.OrthogonalComponents, ...
            Kernel=options.Kernel, Gamma=options.Gamma, Degree=options.Degree, ...
            Offset=options.Offset, Classifier=options.Classifier, Fit=false, ...
            ByColumn=options.ByColumn, Selection=options.Selection, ...
            Oversample=options.Oversample, Power=options.Power, ...
            Folds=plans{run}.Inner{fold});
        selected(fold) = inner.BestNumComponents;
        model = makeModel(inner.BestNumComponents, options, classification, size(X, 2), ...
            options.Seed + 20000 * (run - 1) + fold);
        model.fit(X(train, :), Y(train, :));
        prediction(test, :) = model.predict(X(test, :));
        testResponse = double(model.predictResponses(X(test, :)));
        trainResponse = double(model.predictResponses(X(train, :)));
        if classification
            active = string(model.Classes(:));
            testTarget = double(string(Y(test)) == active');
            trainTarget = double(string(Y(train)) == active');
            proportions = mean(trainTarget, 1);
        else
            testTarget = double(Y(test, :));
            trainTarget = double(Y(train, :));
            proportions = mean(trainTarget, 1);
        end
        q2Press = q2Press + sum((testTarget - testResponse) .^ 2, "all");
        q2Reference = q2Reference + sum((testTarget - proportions) .^ 2, "all");
        trainingEvaluation = fastpls.evaluate(trainTarget, trainResponse, ByColumn=false);
        trainingR2(fold) = trainingEvaluation.R2;
    end
    evaluation = fastpls.evaluate(Y, prediction, ByColumn=options.ByColumn);
    selection = lower(string(options.Selection));
    if selection == "auto"
        if classification, selection = "accuracy"; else, selection = "rmsd"; end
    end
    if q2Reference > 0, q2 = 1 - q2Press / q2Reference; else, q2 = NaN; end
    r2 = mean(trainingR2, "omitnan");
    current = struct( ...
        "Prediction", prediction, ...
        "Fold", outer, ...
        "BestNumComponents", selected, ...
        "SelectionMetric", selection, ...
        "MetricValue", metricValue(selection, evaluation, q2, r2), ...
        "Q2Y", q2, ...
        "R2Y", r2, ...
        "Metrics", evaluation);
    if run == 1
        runs = repmat(current, options.Runs, 1);
    else
        runs(run) = current;
    end
    predictions{run} = prediction;
end
if classification
    values = [predictions{:}];
    combined = strings(size(Y));
    for row = 1:size(values, 1), combined(row) = mode(categorical(values(row, :))); end
else
    stacked = cat(ndims(Y) + 1, predictions{:});
    combined = mean(stacked, ndims(Y) + 1);
end
end

function model = makeModel(component, options, classification, predictors, seed)
classifier = string(options.Classifier);
if classification && strlength(classifier) == 0, classifier = "lda"; end
gamma = options.Gamma;
if isempty(gamma), gamma = 1 / predictors; end
model = fastpls.Model( ...
    NumComponents=component, Method=options.Method, Classifier=classifier, ...
    Scaling=options.Scaling, Backend=options.Backend, ...
    Oversample=options.Oversample, Power=options.Power, Seed=seed, ...
    OrthogonalComponents=options.OrthogonalComponents, Kernel=options.Kernel, ...
    Gamma=gamma, Degree=options.Degree, Offset=options.Offset);
end

function value = metricValue(selection, evaluation, q2, r2)
switch selection
    case "q2y", value = q2;
    case "r2y", value = r2;
    case "accuracy", value = evaluation.Accuracy;
    case "balanced_accuracy", value = evaluation.BalancedAccuracy;
    case "lift_accuracy", value = evaluation.LiftAccuracy;
    case "macro_precision", value = evaluation.MacroPrecision;
    case "macro_recall", value = evaluation.MacroRecall;
    case "macro_f1", value = evaluation.MacroF1;
    case "kappa", value = evaluation.Kappa;
    case "rmsd", value = evaluation.RMSD;
    case "mae", value = evaluation.MAE;
    case "mape_percent", value = evaluation.MAPEPercent;
    case "rpd", value = evaluation.RPD;
    case "pearson_r", value = evaluation.PearsonR;
    case "spearman_r", value = evaluation.SpearmanR;
    otherwise, error("fastPLS:InvalidSelection", "Unknown selection metric.");
end
end

function index = permutationIndices(groups, stream)
[~, ~, inverse] = unique(groups, "stable");
members = arrayfun(@(group) find(inverse == group), 1:max(inverse), UniformOutput=false);
index = (1:numel(groups))';
sizes = unique(cellfun(@numel, members));
for sizeValue = sizes(:)'
    eligible = find(cellfun(@numel, members) == sizeValue);
    donorOrder = eligible(randperm(stream, numel(eligible)));
    for position = 1:numel(eligible)
        index(members{eligible(position)}) = members{donorOrder(position)};
    end
end
end

function folds = makeFolds(sampleCount, kfold, groups, labels, seed)
if numel(groups) ~= sampleCount
    error("fastPLS:InvalidGroups", ...
        "Constrain must contain one group per sample.");
end
[~, first, inverse] = unique(groups, "stable");
groupCount = numel(first);
if isstring(kfold) || ischar(kfold)
    if lower(string(kfold)) ~= "loocv"
        error("fastPLS:InvalidFolds", "Unknown fold specification.");
    end
    foldCount = groupCount;
else
    if ~isscalar(kfold) || ~isfinite(kfold) || kfold < 1 || fix(kfold) ~= kfold
        error("fastPLS:InvalidFolds", ...
            "Fold count must be a positive integer or 'loocv'.");
    end
    foldCount = min(kfold, groupCount);
end
if foldCount < 2, error("fastPLS:InvalidFolds", "At least two folds are required."); end
stream = RandStream("mt19937ar", Seed=seed);
groupFolds = zeros(groupCount, 1);
if foldCount == groupCount
    groupFolds = (1:groupCount)';
elseif isempty(labels)
    order = randperm(stream, groupCount);
    groupFolds(order) = mod(0:(groupCount - 1), foldCount) + 1;
else
    labels = string(labels(:));
    groupLabels = strings(groupCount, 1);
    for group = 1:groupCount
        memberLabels = labels(inverse == group);
        values = unique(memberLabels, "stable");
        counts = arrayfun(@(label) sum(memberLabels == label), values);
        [~, majority] = max(counts);
        groupLabels(group) = values(majority);
    end
    for label = unique(groupLabels, "stable")'
        members = find(groupLabels == label);
        order = members(randperm(stream, numel(members)));
        groupFolds(order) = mod(0:(numel(members) - 1), foldCount) + 1;
    end
end
folds = groupFolds(inverse);
if numel(unique(folds)) ~= foldCount
    error("fastPLS:InvalidFolds", ...
        "Stratified grouped folds cannot populate every fold.");
end
for label = unique(labels, "stable")'
    if numel(unique(folds(labels == label))) < 2
        error("fastPLS:InvalidFolds", ...
            "Each class must occur in at least two folds.");
    end
end
end

function value = isClassification(response, classifier)
value = strlength(string(classifier)) > 0 || iscategorical(response) || ...
    isstring(response) || iscellstr(response) || islogical(response);
end

function matrix = numericMatrix(matrix, name)
if ~isnumeric(matrix) || ~ismatrix(matrix) || ~isreal(matrix) || any(~isfinite(matrix), "all")
    error("fastPLS:InvalidMatrix", "%s must be a finite real matrix.", name);
end
if isvector(matrix), matrix = matrix(:); end
if ~isa(matrix, "single") && ~isa(matrix, "double"), matrix = double(matrix); end
end

function value = conditional(test, yes, no)
if test, value = yes; else, value = no; end
end
