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
metrics = fastpls.evaluate(Y, Y);
assert(abs(metrics.R2 - 1) < eps && metrics.RMSD == 0);
convenience = fastpls.pls(X(1:90, :), Y(1:90, :), X(91:end, :), ...
    NumComponents=2);
assert(isa(convenience.Model, "fastpls.Model"));
assert(isequal(size(convenience.Prediction), [30, 2]));
lowRank = single(reshape(1:40, 20, 2) * reshape(linspace(-1, 1, 16), 2, 8));
[U, D, V] = fastpls.fastsvd(lowRank, 2, Seed=7);
assert(norm(U * diag(D) * V' - lowRank, "fro") / norm(lowRank, "fro") < 2e-4);
assert(norm(fastpls.fastcor(X) - corrcoef(X), "fro") < 1e-10);
fprintf("fastPLS-matlab tests passed.\n");
end
