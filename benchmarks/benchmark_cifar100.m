function report = benchmark_cifar100(dataDirectory, repetitions)
%BENCHMARK_CIFAR100 Benchmark 99-component SIMPLS-LDA on DINOv2 features.
arguments
    dataDirectory = "/tmp"
    repetitions (1,1) double {mustBeInteger,mustBePositive} = 7
end
root = fileparts(fileparts(mfilename("fullpath")));
addpath(root);
rowsTrain = 50000;
rowsTest = 10000;
columns = 768;
Xtrain = readBinary(fullfile(dataDirectory, "cifar100_Xtrain_f64.bin"), ...
    rowsTrain, columns, "double");
Xtest = readBinary(fullfile(dataDirectory, "cifar100_Xtest_f64.bin"), ...
    rowsTest, columns, "double");
ytrain = categorical(readLabels(fullfile(dataDirectory, ...
    "cifar100_ytrain_i32.bin"), rowsTrain));
ytest = categorical(readLabels(fullfile(dataDirectory, ...
    "cifar100_ytest_i32.bin"), rowsTest));
records = struct([]);
for precision = ["single", "double"]
    train = cast(Xtrain, precision);
    test = cast(Xtest, precision);
    elapsed = zeros(repetitions, 1);
    predictions = [];
    for iteration = 1:repetitions
        started = tic;
        model = fastpls.Model(NumComponents=99, Method="simpls", ...
            Classifier="lda", Seed=1);
        model.fit(train, ytrain);
        predictions = model.predict(test);
        elapsed(iteration) = toc(started);
    end
    records(end + 1).precision = precision; %#ok<AGROW>
    records(end).medianSeconds = median(elapsed);
    records(end).iqrSeconds = iqr(elapsed);
    records(end).accuracy = mean(predictions == ytest);
    records(end).replicateSeconds = elapsed';
end
report = struct("dataset", "CIFAR-100 DINOv2 embeddings", ...
    "components", 99, "method", "simpls", "classifier", "lda", ...
    "backend", fastpls.backendInfo(), "repetitions", repetitions, ...
    "records", records);
fprintf("%s\n", jsonencode(report, PrettyPrint=true));
end

function values = readBinary(path, rows, columns, precision)
file = fopen(path, "rb");
if file < 0, error("fastPLS:MissingBenchmarkData", "Cannot open %s.", path); end
cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
values = fread(file, [rows, columns], "*" + precision);
if ~isequal(size(values), [rows, columns])
    error("fastPLS:InvalidBenchmarkData", "Unexpected dimensions in %s.", path);
end
end

function values = readLabels(path, rows)
file = fopen(path, "rb");
if file < 0, error("fastPLS:MissingBenchmarkData", "Cannot open %s.", path); end
cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
values = fread(file, [rows, 1], "*int32");
if numel(values) ~= rows
    error("fastPLS:InvalidBenchmarkData", "Unexpected label count in %s.", path);
end
end
