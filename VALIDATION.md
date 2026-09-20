# Validation record for fastPLS-matlab 0.2.0

## Evaluated source

- Wrapper version: 0.2.0
- Shared fastPLS R core version: 0.3
- Shared core commit: `91811067223ee9b440e8269b6d6957869a0c04b0`
- Platform: macOS arm64
- MATLAB: R2026a
- CPU linear algebra: Apple Accelerate

`tools/check_core.py` confirms that the vendored headers are identical to the
recorded R source.

## Automated tests

The MEX module compiled successfully. `tests/run_tests.m` passed all tests, and
MATLAB Code Analyzer reported zero findings. Coverage includes all four PLS
families, single and double precision, regression, argmax and LDA
classification, ranked prediction, independent-test evaluation, single and
nested cross-validation, grouped permutation testing, rSVD, correlation, and
VIP output. The suite also verifies intercept-only prediction and effective
component diagnostics when a SIMPLS-family fit has zero effective directions.

## R interface agreement

The double-precision record is in
`benchmarks/parity_macos_arm64_v0.2.0.json`; the single-precision record is in
`benchmarks/parity_macos_arm64_v0.2.0_single.json`.

| Family | Double maximum difference | Single maximum difference | LDA agreement |
|---|---:|---:|---:|
| SIMPLS | 2.91e-16 | 5.68e-08 | 100% |
| PLS-SVD | 4.91e-16 | 9.50e-08 | 100% |
| OPLS | 4.13e-16 | 3.46e-07 | 100% |
| Kernel PLS | 7.63e-17 | 1.12e-08 | 100% |

These results establish agreement for the recorded arrays and controls. They
do not establish equivalence between rSVD and a full decomposition or replace
validation on other operating systems.

## Current limitations

The MATLAB package exposes the complete portable CPU workflow. CUDA and Metal
adapters are not included in version 0.2.0; requesting either backend fails
explicitly rather than changing to CPU. The interface is tested with MATLAB
R2026a. GNU Octave compatibility is not claimed.
