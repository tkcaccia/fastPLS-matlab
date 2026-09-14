function report = compare_r(rLibrary, repetitions)
%COMPARE_R Compare MATLAB and R wrappers under identical controls.
arguments
    rLibrary = "/tmp/fastPLS-r-lib"
    repetitions = 11
end
root = fileparts(fileparts(mfilename("fullpath")));
temporary = string(tempname);
mkdir(temporary);
finish = onCleanup(@() rmdir(temporary, "s")); %#ok<NASGU>
rows = (1:2400)';
columns = 1:240;
X = sin(rows .* columns * 0.0017) + cos(rows .* (columns + 2) * 0.0011);
classIndex = mod((0:2399)', 8);
for category = 0:7
    X(classIndex == category, (4 * category + 1):(4 * category + 4)) = ...
        X(classIndex == category, (4 * category + 1):(4 * category + 4)) + 2;
end
coefficients = sin((1:240)' .* (1:32) * 0.013);
Y = X * coefficients / size(X, 2);
Xtrain = X(1:1800, :);
Xtest = X(1801:end, :);
Ytrain = Y(1:1800, :);
labels = "class-" + string(classIndex(1:1800));
writematrix(Xtrain, fullfile(temporary, "Xtrain.csv"));
writematrix(Xtest, fullfile(temporary, "Xtest.csv"));
writematrix(Ytrain, fullfile(temporary, "Ytrain.csv"));
writelines(labels, fullfile(temporary, "labels.txt"));
script = fullfile(root, "benchmarks", "compare_r.R");
command = sprintf('Rscript "%s" "%s" "%s" "%s" %d', ...
    script, temporary, temporary, rLibrary, repetitions);
[status, output] = system(command);
if status ~= 0, error("fastPLS:RBenchmark", "%s", output); end
records = struct([]);
for method = ["simpls", "plssvd", "opls", "kernelpls"]
    elapsed = zeros(repetitions, 1);
    for iteration = 1:repetitions
        started = tic;
        fit = fastpls.Model(NumComponents=8, Method=method, Seed=17, ...
            Kernel="rbf", Gamma=0.1, OrthogonalComponents=1);
        fit.fit(Xtrain, Ytrain);
        prediction = fit.predict(Xtest);
        elapsed(iteration) = toc(started);
    end
    rPrediction = readmatrix(fullfile(temporary, "r_" + method + "_regression.csv"), ...
        NumHeaderLines=1);
    classification = fastpls.Model(NumComponents=8, Method=method, ...
        Classifier="lda", Seed=17, Kernel="rbf", Gamma=0.1, ...
        OrthogonalComponents=1);
    classification.fit(Xtrain, labels);
    matlabLabels = classification.predict(Xtest);
    rLabels = string(readlines(fullfile(temporary, "r_" + method + "_labels.txt")));
    rLabels(rLabels == "") = [];
    difference = prediction - rPrediction;
    records(end + 1).method = method; %#ok<AGROW>
    records(end).maxAbsPredictionDifference = max(abs(difference), [], "all");
    records(end).relativePredictionError = norm(difference, "fro") / norm(rPrediction, "fro");
    records(end).classificationAgreement = mean(matlabLabels == rLabels);
    records(end).matlabMedianSeconds = median(elapsed);
    records(end).rSeconds = str2double(strtrim(fileread(fullfile(temporary, ...
        "r_" + method + "_seconds.txt"))));
end
report = struct("fastPLSRVersion", "0.99.66", ...
    "coreCommit", "3854e369c1f0cd615e58b968b7721efb4b2a3146", ...
    "repetitions", repetitions, "records", records);
fprintf("%s\n", jsonencode(report, PrettyPrint=true));
end
