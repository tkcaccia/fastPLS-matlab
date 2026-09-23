function run_tests()
%RUN_TESTS Deterministic API coverage for fastPLS-matlab.
root = fileparts(fileparts(mfilename("fullpath")));
addpath(root);
rng(42);
X = randn(120, 12);
Y = [X(:, 1) + X(:, 2), X(:, 3) - X(:, 4)];
labels = categorical(X(:, 1) + 0.3 * X(:, 2) > 0);
families = ["simpls", "plssvd", "opls", "kernelpls"];
precisions = ["single", "double"];
for precision = precisions
    values = cast(X, precision);
    responses = cast(Y, precision);
    for family = families
        regression = fastpls.Model(NumComponents=2, Method=family, ...
            Seed=9, Kernel="rbf", Gamma=0.1);
        regression.fit(values(1:90, :), responses(1:90, :));
        prediction = regression.predict(values(91:end, :));
        assert(isequal(size(prediction), [30, 2]));
        assert(isa(prediction, precision));
        assert(all(isfinite(prediction), "all"));
        for classifier = ["argmax", "lda"]
            classification = fastpls.Model(NumComponents=2, Method=family, ...
                Classifier=classifier, Seed=9, Kernel="polynomial", Gamma=0.1);
            classification.fit(values(1:90, :), labels(1:90));
            first = classification.predict(values(91:end, :));
            ranked = classification.predict(values(91:end, :), Top=2);
            assert(isequal(size(first), [30, 1]));
            assert(isequal(size(ranked), [30, 2]));
            assert(all(first == ranked(:, 1)));
        end
    end
end
failed = false;
try
    unavailable = fastpls.Model(Backend="cuda");
    unavailable.fit(X, Y);
catch error
    failed = contains(error.identifier, "UnavailableBackend");
end
assert(failed, "An unavailable backend did not fail explicitly.");
cuda = fastpls.cudaInfo();
assert(cuda.Status == "unavailable" && ~cuda.Compiled && ...
    ~cuda.Available && ~cuda.DiagnosticOnly && cuda.DeviceCount == 0 && ...
    ismissing(cuda.RuntimeVersion) && ismissing(cuda.DriverVersion) && ...
    cuda.NoCpuFallback);
defaultClassification = fastpls.Model(NumComponents=2, Seed=31);
defaultClassification.fit(X(1:90, :), labels(1:90));
explicitClassification = fastpls.Model( ...
    NumComponents=2, Classifier="lda", Seed=31);
explicitClassification.fit(X(1:90, :), labels(1:90));
assert(all(defaultClassification.predict(X(91:end, :)) == ...
    explicitClassification.predict(X(91:end, :))));
metrics = fastpls.evaluate(Y, Y);
assert(abs(metrics.R2 - 1) < eps && metrics.RMSD == 0);
integerRegression = fastpls.evaluate([1; 2; 3], [1; 2; 3]);
assert(integerRegression.Task == "regression");
shifted = [0, 100; 1, 101; 2, 102];
rpd = fastpls.evaluate(shifted, shifted + 1, ByColumn=false);
centered = shifted - mean(shifted, 1);
expectedRPD = sqrt(sum(centered .^ 2, "all") / (numel(shifted) - 1));
assert(abs(rpd.RPD - expectedRPD) < 10 * eps);
defaultModel = fastpls.Model(NumComponents=2, Seed=19);
defaultModel.fit(X, Y);
centeredModel = fastpls.Model( ...
    NumComponents=2, Scaling="centering", Seed=19);
centeredModel.fit(X, Y);
assert(isequal(defaultModel.predict(X), centeredModel.predict(X)));
for precision = precisions
    constantX = cast(randn(41, 30), precision);
    for constantValue = [0, 2.5]
        constantModel = fastpls.Model(NumComponents=10, Seed=20261542);
        constantModel.fit(constantX, ...
            cast(repmat(constantValue, 41, 1), precision));
        constantPrediction = constantModel.predict(constantX(1:7, :));
        assert(constantModel.NumComponents == 10);
        assert(constantModel.NumComponentsFitted == 0);
        assert(isequal(size(constantPrediction), [7, 1]));
        assert(all(constantPrediction == cast(constantValue, precision), "all"));
    end
end
defaultKernel = fastpls.Model( ...
    NumComponents=2, Method="kernelpls", Kernel="rbf", Seed=23);
defaultKernel.fit(X, Y);
explicitKernel = fastpls.Model( ...
    NumComponents=2, Method="kernelpls", Kernel="rbf", ...
    Gamma=1 / size(X, 2), Seed=23);
explicitKernel.fit(X, Y);
assert(isequal(defaultKernel.predict(X), explicitKernel.predict(X)));
for invalidComponent = [0, 1.5]
    failed = false;
    try
        invalid = fastpls.Model(NumComponents=invalidComponent);
        invalid.fit(X, Y);
    catch error
        failed = contains(error.identifier, "InvalidControl");
    end
    assert(failed, "Invalid component control was accepted.");
end
invalidModels = {
    fastpls.Model(Method="unknown"), ...
    fastpls.Model(Classifier="knn"), ...
    fastpls.Model(Scaling="unit"), ...
    fastpls.Model(Kernel="sigmoid")
};
for controlIndex = 1:numel(invalidModels)
    failed = false;
    try
        invalid = invalidModels{controlIndex};
        invalid.fit(X, Y);
    catch
        failed = true;
    end
    assert(failed, "An unsupported model control was accepted.");
end
convenience = fastpls.pls(X(1:90, :), Y(1:90, :), X(91:end, :), ...
    NumComponents=2, YTest=Y(91:end, :));
assert(isa(convenience.Model, "fastpls.Model"));
assert(isequal(size(convenience.Prediction), [30, 2]));
assert(convenience.Metrics.Task == "regression");
assert(isfinite(convenience.Metrics.Q2));
numericLabels = double(labels);
numericClassification = fastpls.pls( ...
    X(1:90, :), numericLabels(1:90), X(91:end, :), ...
    NumComponents=2, Classifier="lda", ...
    YTest=numericLabels(91:end));
assert(numericClassification.Metrics.Task == "classification");
stored = fastpls.Model(NumComponents=2, StoreScores=true);
stored.fit(X, Y);
importance = fastpls.vip(stored);
assert(iscell(importance) && numel(importance) == size(Y, 2));
for family = families
    arguments = {"NumComponents", [1, 2], "KFold", 3, ...
        "Method", family, "Classifier", "lda", ...
        "Selection", "balanced_accuracy", "Fit", false, "Seed", 11};
    if family == "kernelpls"
        arguments = [arguments, {"Kernel", "rbf", "Gamma", 0.1}]; %#ok<AGROW>
    end
    selected = fastpls.plsSingleCV(X, string(labels), arguments{:});
    assert(ismember(selected.BestNumComponents, [1, 2]));
    assert(isfinite(selected.BestMetricValue));
end
nested = fastpls.plsDoubleCV(X, Y, NumComponents=[1, 2], ...
    InnerFolds=2, OuterFolds=2, Selection="Q2Y", Seed=13);
assert(isfinite(nested.Q2Y) && isfinite(nested.R2Y));
singlePrecisionCV = fastpls.plsSingleCV(single(X), single(Y), ...
    NumComponents=[1, 2], KFold=3, Selection="RMSD", Fit=false, Seed=13);
assert(isfinite(singlePrecisionCV.BestMetricValue));
permuted = fastpls.plsDoubleCV(X, string(labels), NumComponents=[1, 2], ...
    InnerFolds=2, OuterFolds=2, Classifier="lda", ...
    Selection="balanced_accuracy", PermutationTest=true, Times=2, Seed=13);
assert(permuted.PValue > 0 && permuted.PValue <= 1);
plotResult = struct("PermutationSampled", [0.2; 0.3], ...
    "PermutationObserved", 0.8, "PermutationMetric", "accuracy");
permutationFigure = figure(Visible="off");
permutationAxes = axes(permutationFigure);
returnedAxes = fastpls.plotPermutation(plotResult, Parent=permutationAxes);
assert(returnedAxes == permutationAxes);
close(permutationFigure);
lowRank = single(reshape(1:40, 20, 2) * reshape(linspace(-1, 1, 16), 2, 8));
[U, D, V] = fastpls.fastsvd(lowRank, 2, Seed=7);
assert(norm(U * diag(D) * V' - lowRank, "fro") / norm(lowRank, "fro") < 2e-4);
assert(norm(fastpls.fastcor(X) - corrcoef(X') , "fro") < 1e-10);
assert(norm(fastpls.fastcor(X, ByRow=false) - corrcoef(X), "fro") < 1e-10);
crossCorrelation = corrcoef(X(1:8, :)');
assert(norm(fastpls.fastcor(X(1:4, :), X(5:8, :), Diagonal=false) - ...
    crossCorrelation(1:4, 5:8), "fro") < 1e-10);
fprintf("fastPLS-matlab tests passed.\n");
end
