# fastPLS-matlab

`fastPLS-matlab` is a MATLAB interface to the same MIT-licensed C++17 core
used by the [`fastPLS`](https://github.com/tkcaccia/fastPLS) R package. It
supports `single` and `double` execution for SIMPLS, PLS-SVD, OPLS, and linear,
radial-basis, or polynomial kernel PLS. Classification uses argmax or pooled-
covariance LDA prediction heads.

The core snapshot is pinned in `UPSTREAM_CORE.json`. Run
`tools/check_core.py /path/to/fastPLS` to verify that every vendored header is
identical to the recorded R-package source.

## Build

MATLAB R2026a on Apple silicon:

```matlab
cd('/path/to/fastPLS-matlab')
build_fastpls
```

The macOS build links Apple Accelerate. Linux builds require OpenBLAS. A C++17
MEX compiler is required.

## Use

```matlab
rng(7)
X = single(randn(200, 30));
y = X(:, 1) - 0.5 * X(:, 2);

model = fastpls.Model(NumComponents=3, Method="simpls", Seed=7);
model.fit(X(1:150, :), y(1:150));
prediction = model.predict(X(151:end, :));
metrics = fastpls.evaluate(y(151:end), prediction);
```

For classification, set `Classifier="argmax"` or `Classifier="lda"`.
`model.predict(X, Top=5)` returns five ranked labels per observation.

## Validation

```matlab
addpath('tests')
run_tests

addpath('benchmarks')
compare_r('/tmp/fastPLS-r-lib', 11)
```

On the macOS arm64 validation run, all four MATLAB estimators were compared
with fastPLS R under identical controls. The maximum absolute prediction
difference was `1.55e-15`, and LDA labels agreed for every held-out sample.
CIFAR-100 SIMPLS-LDA with 99 components took 0.208 seconds in `single` and
0.281 seconds in `double`; accuracy was 0.8687 in both precisions. The matched
R medians were 0.197 and 0.365 seconds, respectively.

The CIFAR-100 result can be regenerated from the same column-major binary
matrices used by the R benchmark:

```matlab
addpath('benchmarks')
benchmark_cifar100('/path/to/cifar100/binaries', 7)
```

## Backend status

Version 0.1.0 validates the shared CPU core. CUDA and Metal requests fail
explicitly rather than silently falling back to CPU. MATLAB accelerator
adapters are not part of this initial release.

## License

MIT. The vendored core and MATLAB interface are MIT licensed.
