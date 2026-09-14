classdef Model < handle
    %MODEL Partial least-squares estimator backed by the fastPLS C++ core.
    properties
        NumComponents = 2
        Method = "simpls"
        Classifier = ""
        Scaling = "autoscaling"
        Backend = "cpu"
        Oversample = 32
        Power = 5
        Seed = 1
        OrthogonalComponents = 1
        Kernel = "linear"
        Gamma = 1
        Degree = 3
        Offset = 1
        StoreScores = false
    end
    properties (SetAccess = private)
        Classes = []
        NumComponentsFitted = 0
        Precision = ""
    end
    properties (Access = private)
        NativeHandle = uint64(0)
        NumPredictors = 0
    end
    methods
        function obj = Model(options)
            arguments
                options.NumComponents = 2
                options.Method = "simpls"
                options.Classifier = ""
                options.Scaling = "autoscaling"
                options.Backend = "cpu"
                options.Oversample = 32
                options.Power = 5
                options.Seed = 1
                options.OrthogonalComponents = 1
                options.Kernel = "linear"
                options.Gamma = 1
                options.Degree = 3
                options.Offset = 1
                options.StoreScores = false
            end
            names = fieldnames(options);
            for index = 1:numel(names)
                obj.(names{index}) = options.(names{index});
            end
        end
        function fit(obj, X, Y)
            obj.release();
            if lower(string(obj.Backend)) ~= "cpu"
                error("fastPLS:UnavailableBackend", ...
                    "CUDA and Metal are not silently replaced by CPU.");
            end
            X = obj.numericMatrix(X, "X");
            obj.NumPredictors = size(X, 2);
            obj.Precision = string(class(X));
            if strlength(string(obj.Classifier)) > 0
                if ~isvector(Y), error("fastPLS:InvalidLabels", "Labels must be a vector."); end
                [obj.Classes, ~, encoded] = unique(Y(:), "stable");
                Ynative = int32(encoded - 1);
                classifier = lower(string(obj.Classifier));
            else
                obj.Classes = [];
                Ynative = cast(obj.numericMatrix(Y, "Y"), "like", X);
                classifier = "regression";
            end
            if size(Ynative, 1) ~= size(X, 1)
                error("fastPLS:DimensionMismatch", "X and Y rows differ.");
            end
            controls = struct("components", obj.NumComponents, ...
                "method", char(lower(string(obj.Method))), ...
                "classifier", char(classifier), ...
                "scaling", char(lower(string(obj.Scaling))), ...
                "oversample", obj.Oversample, "power", obj.Power, ...
                "seed", obj.Seed, ...
                "orthogonalComponents", obj.OrthogonalComponents, ...
                "kernel", char(lower(string(obj.Kernel))), ...
                "gamma", obj.Gamma, "degree", obj.Degree, ...
                "offset", obj.Offset, "storeScores", obj.StoreScores);
            [obj.NativeHandle, obj.NumComponentsFitted] = ...
                fastpls_mex('new', X, Ynative, controls);
        end
        function prediction = predict(obj, X, options)
            arguments
                obj
                X
                options.Top = []
            end
            X = obj.predictionMatrix(X);
            if isempty(obj.Classes)
                if ~isempty(options.Top), error("fastPLS:InvalidTop", "Top is classification-only."); end
                prediction = fastpls_mex('predict', obj.handle(), X);
            else
                top = options.Top;
                if isempty(top), top = 1; end
                indices = fastpls_mex('classes', obj.handle(), X, top);
                prediction = obj.Classes(indices);
                if top == 1, prediction = prediction(:, 1); end
            end
        end
        function values = predictScores(obj, X)
            values = fastpls_mex('scores', obj.handle(), obj.predictionMatrix(X));
        end
        function delete(obj), obj.release(); end
    end
    methods (Access = private)
        function value = handle(obj)
            if obj.NativeHandle == 0, error("fastPLS:NotFitted", "Call fit first."); end
            value = obj.NativeHandle;
        end
        function release(obj)
            if obj.NativeHandle ~= 0
                fastpls_mex('delete', obj.NativeHandle);
                obj.NativeHandle = uint64(0);
            end
        end
        function X = predictionMatrix(obj, X)
            X = cast(obj.numericMatrix(X, "X"), obj.Precision);
            if size(X, 2) ~= obj.NumPredictors
                error("fastPLS:DimensionMismatch", "Predictor counts differ.");
            end
        end
    end
    methods (Static, Access = private)
        function X = numericMatrix(X, name)
            if ~isnumeric(X) || ~ismatrix(X) || ~isreal(X) || any(~isfinite(X), "all")
                error("fastPLS:InvalidMatrix", "%s must be a finite real matrix.", name);
            end
            if ~isa(X, "single") && ~isa(X, "double"), X = double(X); end
        end
    end
end
