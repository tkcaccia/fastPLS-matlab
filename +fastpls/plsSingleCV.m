function result = plsSingleCV(Xdata, Ydata, options)
%PLSSINGLECV Select a component count by grouped cross-validation.
arguments
    Xdata {mustBeNumeric,mustBeReal}
    Ydata
    options.NumComponents = 2
    options.Constrain = []
    options.Scaling = "centering"
    options.Method = "simpls"
    options.Backend = "cpu"
    options.Seed = 1
    options.KFold = 10
    options.OrthogonalComponents = 1
    options.Kernel = "linear"
    options.Gamma = []
    options.Degree = 3
    options.Offset = 1
    options.Classifier = ""
    options.Fit (1, 1) logical = true
    options.ByColumn (1, 1) logical = false
    options.Selection = "auto"
    options.Oversample = 32
    options.Power = 5
    options.Folds = []
end
if lower(string(options.Backend)) ~= "cpu"
    error("fastPLS:UnavailableBackend", ...
        "fastPLS-matlab currently provides only the CPU backend.");
end
Xdata = numericMatrix(Xdata, "Xdata");
components = unique(double(options.NumComponents(:)'), "sorted");
if isempty(components) || any(components < 1) || any(fix(components) ~= components)
    error("fastPLS:InvalidComponents", ...
        "NumComponents must contain positive integers.");
end
classification = isClassification(Ydata, options.Classifier);
if classification
    Ydata = string(Ydata(:));
    classes = unique(Ydata, "stable");
    classifier = string(options.Classifier);
    if strlength(classifier) == 0, classifier = "lda"; end
else
    Ydata = cast(numericMatrix(Ydata, "Ydata"), "like", Xdata);
    classes = strings(0, 1);
    classifier = "";
end
if size(Ydata, 1) ~= size(Xdata, 1)
    error("fastPLS:DimensionMismatch", "Xdata and Ydata rows differ.");
end
selection = lower(string(options.Selection));
if selection == "auto"
    if classification, selection = "accuracy"; else, selection = "rmsd"; end
end
validateSelection(selection, classification);
if isempty(options.Gamma), gamma = 1 / size(Xdata, 2); else, gamma = options.Gamma; end
if isempty(options.Folds)
    folds = makeFolds(size(Xdata, 1), options.KFold, options.Constrain, ...
        conditional(classification, Ydata, []), options.Seed);
else
    folds = double(options.Folds(:));
    folds = validateFolds(folds, size(Xdata, 1), options.Constrain, ...
        conditional(classification, Ydata, []));
end
predictions = cell(1, numel(components));
crossValidated = cell(1, numel(components));
fitted = cell(1, numel(components));
q2 = nan(1, numel(components));
r2 = nan(1, numel(components));
rmsd = nan(1, numel(components));
selectionValues = nan(1, numel(components));
for componentIndex = 1:numel(components)
    component = components(componentIndex);
    if classification
        prediction = strings(size(Ydata));
        responseScores = zeros(size(Ydata, 1), numel(classes), "like", Xdata);
    else
        prediction = zeros(size(Ydata), "like", Ydata);
    end
    foldReference = 0;
    for fold = unique(folds(:))'
        train = folds ~= fold;
        test = ~train;
        model = makeModel(component, classifier, options, gamma);
        model.fit(Xdata(train, :), Ydata(train, :));
        prediction(test, :) = model.predict(Xdata(test, :));
        score = model.predictResponses(Xdata(test, :));
        if classification
            for classIndex = 1:numel(model.Classes)
                destination = find(classes == string(model.Classes(classIndex)), 1);
                responseScores(test, destination) = score(:, classIndex);
            end
            target = double(Ydata(test) == classes');
            proportions = mean(double(Ydata(train) == classes'), 1);
            foldReference = foldReference + sum((target - proportions) .^ 2, "all");
        else
            foldReference = foldReference + sum( ...
                (double(Ydata(test, :)) - mean(double(Ydata(train, :)), 1)) .^ 2, "all");
        end
    end
    predictions{componentIndex} = prediction;
    crossValidated{componentIndex} = fastpls.evaluate( ...
        Ydata, prediction, ByColumn=options.ByColumn);
    if classification
        target = double(Ydata == classes');
        press = sum((target - double(responseScores)) .^ 2, "all");
    else
        press = sum((double(Ydata) - double(prediction)) .^ 2, "all");
        rmsd(componentIndex) = crossValidated{componentIndex}.RMSD;
    end
    if foldReference > 0, q2(componentIndex) = 1 - press / foldReference; end
    if options.Fit || selection == "r2y"
        fullModel = makeModel(component, classifier, options, gamma);
        fullModel.fit(Xdata, Ydata);
        fullResponse = fullModel.predictResponses(Xdata);
        if classification, target = double(Ydata == classes'); else, target = double(Ydata); end
        fitted{componentIndex} = fastpls.evaluate( ...
            target, double(fullResponse), ByColumn=options.ByColumn);
        r2(componentIndex) = fitted{componentIndex}.R2;
    end
    selectionValues(componentIndex) = metricValue( ...
        selection, crossValidated{componentIndex}, q2(componentIndex), r2(componentIndex));
end
if ismember(selection, ["rmsd", "mae", "mape_percent"])
    [bestValue, bestIndex] = min(selectionValues);
else
    [bestValue, bestIndex] = max(selectionValues);
end
result = struct( ...
    "BestNumComponents", components(bestIndex), ...
    "BestIndex", bestIndex, ...
    "SelectionMetric", selection, ...
    "BestMetricValue", bestValue, ...
    "NumComponents", components, ...
    "Fold", folds, ...
    "Prediction", {predictions}, ...
    "Q2Y", q2, ...
    "R2Y", r2, ...
    "RMSD", rmsd, ...
    "SelectionValues", selectionValues, ...
    "Metrics", struct("CrossValidated", {crossValidated}, "Fitted", {fitted}), ...
    "Method", string(options.Method), ...
    "Backend", string(options.Backend));
end

function model = makeModel(component, classifier, options, gamma)
model = fastpls.Model( ...
    NumComponents=component, Method=options.Method, Classifier=classifier, ...
    Scaling=options.Scaling, Backend=options.Backend, ...
    Oversample=options.Oversample, Power=options.Power, Seed=options.Seed, ...
    OrthogonalComponents=options.OrthogonalComponents, Kernel=options.Kernel, ...
    Gamma=gamma, Degree=options.Degree, Offset=options.Offset);
end

function value = isClassification(response, classifier)
value = strlength(string(classifier)) > 0 || iscategorical(response) || ...
    isstring(response) || iscellstr(response) || islogical(response);
end

function validateSelection(selection, classification)
if classification
    allowed = ["accuracy", "balanced_accuracy", "lift_accuracy", ...
        "macro_precision", "macro_recall", "macro_f1", "kappa", "r2y", "q2y"];
else
    allowed = ["r2y", "q2y", "rmsd", "mae", "mape_percent", ...
        "rpd", "pearson_r", "spearman_r"];
end
if ~ismember(selection, allowed)
    error("fastPLS:InvalidSelection", ...
        "Selection '%s' is not valid for this task.", selection);
end
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
end
end

function folds = makeFolds(sampleCount, kfold, groups, labels, seed)
if isempty(groups), groups = (1:sampleCount)'; else, groups = groups(:); end
if numel(groups) ~= sampleCount
    error("fastPLS:InvalidGroups", "Constrain must contain one group per sample.");
end
[~, first, inverse] = unique(groups, "stable");
groupCount = numel(first);
if isstring(kfold) || ischar(kfold)
    if lower(string(kfold)) ~= "loocv", error("fastPLS:InvalidFolds", "Unknown KFold value."); end
    foldCount = groupCount;
else
    if ~isscalar(kfold) || ~isfinite(kfold) || kfold < 1 || fix(kfold) ~= kfold
        error("fastPLS:InvalidFolds", ...
            "KFold must be a positive integer or 'loocv'.");
    end
    foldCount = min(double(kfold), groupCount);
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

function folds = validateFolds(folds, sampleCount, groups, labels)
if numel(folds) ~= sampleCount || any(~isfinite(folds)) || ...
        any(folds < 1) || any(fix(folds) ~= folds)
    error("fastPLS:InvalidFolds", ...
        "Folds must contain one positive integer per row.");
end
[~, ~, folds] = unique(folds, "stable");
if numel(unique(folds)) < 2
    error("fastPLS:InvalidFolds", "At least two folds are required.");
end
if ~isempty(groups)
    groups = groups(:);
    if numel(groups) ~= sampleCount
        error("fastPLS:InvalidGroups", ...
            "Constrain must contain one group per sample.");
    end
    [~, ~, groupIndex] = unique(groups, "stable");
    for group = 1:max(groupIndex)
        if numel(unique(folds(groupIndex == group))) ~= 1
            error("fastPLS:InvalidGroups", ...
                "Each constrained group must remain in one fold.");
        end
    end
end
if ~isempty(labels)
    labels = string(labels(:));
    for label = unique(labels, "stable")'
        if numel(unique(folds(labels == label))) < 2
            error("fastPLS:InvalidFolds", ...
                "Each class must occur in at least two folds.");
        end
    end
end
folds = double(folds);
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
