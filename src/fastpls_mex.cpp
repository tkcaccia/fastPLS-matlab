// SPDX-License-Identifier: MIT
#include "mex.h"
#include "native_backend.hpp"

#include <fastpls/core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace core = fastpls::core;
using fastpls_matlab::NativeBackend;

namespace {

enum class Family { simpls, plssvd, opls, kernelpls };
enum class Head { regression, argmax, lda };

std::string text_value(const mxArray* value, const char* name) {
  if (value == nullptr || !mxIsChar(value)) {
    throw std::invalid_argument(std::string(name) + " must be text");
  }
  char* raw = mxArrayToUTF8String(value);
  if (raw == nullptr) throw std::runtime_error("could not decode text input");
  std::string result(raw);
  mxFree(raw);
  return result;
}

const mxArray* field(const mxArray* options, const char* name) {
  if (options == nullptr || !mxIsStruct(options)) {
    throw std::invalid_argument("options must be a scalar structure");
  }
  const mxArray* value = mxGetField(options, 0, name);
  if (value == nullptr) {
    throw std::invalid_argument(std::string("missing option: ") + name);
  }
  return value;
}

double number_field(const mxArray* options, const char* name) {
  const mxArray* value = field(options, name);
  if (!mxIsNumeric(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
    throw std::invalid_argument(std::string(name) + " must be a real scalar");
  }
  return mxGetScalar(value);
}

bool bool_field(const mxArray* options, const char* name) {
  const mxArray* value = field(options, name);
  if (mxIsLogicalScalar(value)) return mxIsLogicalScalarTrue(value);
  return number_field(options, name) != 0.0;
}

Family parse_family(const std::string& value) {
  if (value == "simpls") return Family::simpls;
  if (value == "plssvd" || value == "pls-svd") return Family::plssvd;
  if (value == "opls") return Family::opls;
  if (value == "kernelpls" || value == "kernel-pls") return Family::kernelpls;
  throw std::invalid_argument("Method must be simpls, plssvd, opls, or kernelpls");
}

Head parse_head(const std::string& value) {
  if (value == "regression") return Head::regression;
  if (value == "argmax") return Head::argmax;
  if (value == "lda") return Head::lda;
  throw std::invalid_argument("Classifier must be regression, argmax, or lda");
}

core::PredictorScaling parse_scaling(const std::string& value) {
  if (value == "autoscaling" || value == "scale") {
    return core::PredictorScaling::autoscaling;
  }
  if (value == "centering" || value == "center") {
    return core::PredictorScaling::centering;
  }
  if (value == "none") return core::PredictorScaling::none;
  throw std::invalid_argument("Scaling must be autoscaling, centering, or none");
}

core::KernelType parse_kernel(const std::string& value) {
  if (value == "linear") return core::KernelType::linear;
  if (value == "rbf" || value == "radial_basis") {
    return core::KernelType::radial_basis;
  }
  if (value == "polynomial" || value == "poly") {
    return core::KernelType::polynomial;
  }
  throw std::invalid_argument("Kernel must be linear, rbf, or polynomial");
}

template<class T>
core::ConstMatrixView<T> mx_view(const mxArray* value, const char* name) {
  const mxClassID expected = std::is_same<T, float>::value ?
      mxSINGLE_CLASS : mxDOUBLE_CLASS;
  if (value == nullptr || mxGetClassID(value) != expected ||
      mxIsComplex(value) || mxIsSparse(value) || mxGetNumberOfDimensions(value) != 2) {
    throw std::invalid_argument(std::string(name) + " has an invalid numeric type");
  }
  const std::size_t rows = static_cast<std::size_t>(mxGetM(value));
  const std::size_t columns = static_cast<std::size_t>(mxGetN(value));
  return core::ConstMatrixView<T>(
      static_cast<const T*>(mxGetData(value)), rows, columns, rows);
}

template<class T>
core::Matrix<T> copy_matrix(core::ConstMatrixView<T> input) {
  core::Matrix<T> output(input.rows(), input.columns());
  for (std::size_t column = 0; column < input.columns(); ++column) {
    std::copy_n(input.data() + column * input.leading_dimension(), input.rows(),
                output.data() + column * output.rows());
  }
  return output;
}

template<class T>
mxArray* mx_matrix(const core::Matrix<T>& input) {
  const mxClassID type = std::is_same<T, float>::value ?
      mxSINGLE_CLASS : mxDOUBLE_CLASS;
  mxArray* output = mxCreateNumericMatrix(
      static_cast<mwSize>(input.rows()), static_cast<mwSize>(input.columns()),
      type, mxREAL);
  std::memcpy(mxGetData(output), input.data(), input.size() * sizeof(T));
  return output;
}

template<class T>
void standardize(core::Matrix<T>& matrix, const std::vector<T>& center,
                 const std::vector<T>& scale) {
  if (matrix.columns() != center.size() || center.size() != scale.size()) {
    throw std::invalid_argument("prediction columns do not match training");
  }
  for (std::size_t column = 0; column < matrix.columns(); ++column) {
    if (!(scale[column] > T(0))) throw std::runtime_error("invalid predictor scale");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      matrix(row, column) = (matrix(row, column) - center[column]) / scale[column];
    }
  }
}

template<class T>
void standardize_predictor_gram(
    core::MatrixView<T> gram, std::size_t sample_count,
    const std::vector<T>& center, const std::vector<T>& scale,
    core::PredictorScaling scaling) {
  if (scaling == core::PredictorScaling::none) return;
  const T samples = static_cast<T>(sample_count);
  for (std::size_t column = 0; column < gram.columns(); ++column) {
    for (std::size_t row = 0; row < gram.rows(); ++row) {
      gram(row, column) =
          (gram(row, column) - samples * center[row] * center[column]) /
          (scale[row] * scale[column]);
    }
  }
}

core::SimplsControls simpls_controls(
    std::size_t n, std::size_t p, std::size_t q, std::size_t components,
    bool classification, int oversample, int power, unsigned int seed) {
  core::SimplsControls controls;
  controls.components = components;
  controls.maximum_block = core::simpls_candidate_block_size(
      components, p, q, classification, n, 64);
  const bool cache = (components >= 20 || p <= 5 * components) &&
      p <= n && n >= p * 8 && p <= 2048;
  controls.cache_predictor_crossprod = cache;
  controls.batch_candidate_geometry = controls.maximum_block > 1 && !cache;
  controls.reorthogonalize = false;
  controls.store_scores = false;
  controls.store_score_moments = classification;
  controls.use_right_gram = true;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  return controls;
}

template<class T>
using ModelVariant = std::variant<
    core::SimplsModel<T>, core::PlssvdModel<T>,
    core::OplsModel<T>, core::KernelPlsModel<T>>;

class ModelBase {
 public:
  virtual ~ModelBase() = default;
  virtual mxArray* predict(const mxArray* input) const = 0;
  virtual mxArray* scores(const mxArray* input) const = 0;
  virtual mxArray* classes(const mxArray* input, std::size_t top) const = 0;
  virtual std::size_t components() const = 0;
  virtual bool is_single() const = 0;
};

template<class T>
class Model final : public ModelBase {
 public:
  Model(Family family, Head head, std::size_t components,
        std::size_t classes, ModelVariant<T> model,
        std::vector<T> predictor_center, std::vector<T> predictor_scale,
        std::vector<T> response_mean, core::LdaModel<T> lda,
        core::Matrix<T> direct_weights, std::vector<T> direct_offsets)
      : family_(family), head_(head), components_(components), classes_(classes),
        model_(std::move(model)), predictor_center_(std::move(predictor_center)),
        predictor_scale_(std::move(predictor_scale)),
        response_mean_(std::move(response_mean)), lda_(std::move(lda)),
        direct_weights_(std::move(direct_weights)),
        direct_offsets_(std::move(direct_offsets)) {}

  mxArray* predict(const mxArray* input) const override {
    return mx_matrix(predict_response(copy_matrix(mx_view<T>(input, "X"))));
  }

  mxArray* scores(const mxArray* input) const override {
    return mx_matrix(latent_scores(copy_matrix(mx_view<T>(input, "X"))));
  }

  mxArray* classes(const mxArray* input, std::size_t top) const override {
    if (head_ == Head::regression) {
      throw std::invalid_argument("ranked prediction is unavailable for regression");
    }
    if (top < 1 || top > classes_) {
      throw std::invalid_argument("Top exceeds the number of classes");
    }
    const auto input_view = mx_view<T>(input, "X");
    core::Matrix<T> values;
    NativeBackend<T> backend;
    if (direct_weights_.size() != 0) {
      values.resize(input_view.rows(), classes_);
      backend.gemm(input_view, direct_weights_.view(), false, false, values.view());
      for (std::size_t column = 0; column < classes_; ++column) {
        for (std::size_t row = 0; row < values.rows(); ++row) {
          values(row, column) += direct_offsets_[column];
        }
      }
    } else if (head_ == Head::lda) {
      values = core::lda_scores(
          latent_scores(copy_matrix(input_view)).view(), lda_);
    } else {
      values = predict_response(copy_matrix(input_view));
    }
    mxArray* output = mxCreateNumericMatrix(
        static_cast<mwSize>(values.rows()), static_cast<mwSize>(top),
        mxINT64_CLASS, mxREAL);
    auto* destination = static_cast<std::int64_t*>(mxGetData(output));
    std::vector<std::size_t> order(classes_);
    for (std::size_t row = 0; row < values.rows(); ++row) {
      for (std::size_t index = 0; index < classes_; ++index) order[index] = index;
      std::partial_sort(order.begin(), order.begin() + top, order.end(),
                        [&](std::size_t left, std::size_t right) {
        return values(row, left) > values(row, right);
      });
      for (std::size_t rank = 0; rank < top; ++rank) {
        destination[row + rank * values.rows()] =
            static_cast<std::int64_t>(order[rank] + 1);
      }
    }
    return output;
  }

  std::size_t components() const override { return components_; }
  bool is_single() const override { return std::is_same<T, float>::value; }

 private:
  core::Matrix<T> latent_scores(core::Matrix<T> predictors) const {
    NativeBackend<T> backend;
    if (family_ == Family::simpls || family_ == Family::plssvd) {
      standardize(predictors, predictor_center_, predictor_scale_);
      const auto& weights = family_ == Family::simpls ?
          std::get<core::SimplsModel<T>>(model_).weights :
          std::get<core::PlssvdModel<T>>(model_).weights;
      core::ConstMatrixView<T> prefix(
          weights.data(), weights.rows(), components_, weights.rows());
      core::Matrix<T> output(predictors.rows(), components_);
      backend.gemm(predictors.view(), prefix, false, false, output.view());
      return output;
    }
    if (family_ == Family::opls) {
      const auto& model = std::get<core::OplsModel<T>>(model_);
      auto filtered = core::apply_opls_filter(
          std::move(predictors), model.filter.predictor_center.data(),
          model.filter.predictor_scale.data(),
          model.filter.predictor_center.size(), model.filter.weights.view(),
          model.filter.loadings.view(), backend);
      core::Matrix<T> output(filtered.rows(), components_);
      core::ConstMatrixView<T> weights(
          model.inner.weights.data(), model.inner.weights.rows(), components_,
          model.inner.weights.rows());
      backend.gemm(filtered.view(), weights, false, false, output.view());
      return output;
    }
    const auto& model = std::get<core::KernelPlsModel<T>>(model_);
    core::kernelpls_detail::standardize(
        predictors.view(), model.predictor_center, model.predictor_scale);
    core::Matrix<T> design;
    if (model.kernel == core::KernelType::linear) {
      design = std::move(predictors);
    } else {
      design = core::kernel_matrix<T>(
          core::ConstMatrixView<T>(predictors.view()), model.reference.view(),
          model.kernel, model.gamma, model.degree, model.offset, backend);
      core::center_kernel_test(
          design.view(), model.kernel_column_means.data(),
          model.kernel_column_means.size(), model.kernel_grand_mean);
    }
    core::Matrix<T> output(design.rows(), components_);
    core::ConstMatrixView<T> weights(
        model.inner.weights.data(), model.inner.weights.rows(), components_,
        model.inner.weights.rows());
    backend.gemm(design.view(), weights, false, false, output.view());
    return output;
  }

  core::Matrix<T> predict_response(core::Matrix<T> predictors) const {
    NativeBackend<T> backend;
    if (family_ == Family::simpls) {
      standardize(predictors, predictor_center_, predictor_scale_);
      auto output = core::predict_simpls_preprocessed<T>(
          predictors.view(), std::get<core::SimplsModel<T>>(model_),
          components_, backend);
      add_response_mean(output);
      return output;
    }
    if (family_ == Family::plssvd) {
      standardize(predictors, predictor_center_, predictor_scale_);
      const auto& model = std::get<core::PlssvdModel<T>>(model_);
      core::Matrix<T> score(predictors.rows(), components_);
      core::ConstMatrixView<T> weights(
          model.weights.data(), model.weights.rows(), components_, model.weights.rows());
      backend.gemm(predictors.view(), weights, false, false, score.view());
      core::Matrix<T> output(predictors.rows(), response_mean_.size());
      backend.gemm(score.view(), model.prediction_weights.front().view(),
                   false, false, output.view());
      add_response_mean(output);
      return output;
    }
    if (family_ == Family::opls) {
      return core::predict_opls(
          std::get<core::OplsModel<T>>(model_), std::move(predictors),
          components_, backend);
    }
    return core::predict_kernelpls(
        std::get<core::KernelPlsModel<T>>(model_), std::move(predictors),
        components_, backend);
  }

  void add_response_mean(core::Matrix<T>& output) const {
    for (std::size_t column = 0; column < output.columns(); ++column) {
      for (std::size_t row = 0; row < output.rows(); ++row) {
        output(row, column) += response_mean_[column];
      }
    }
  }

  Family family_;
  Head head_;
  std::size_t components_;
  std::size_t classes_;
  ModelVariant<T> model_;
  std::vector<T> predictor_center_;
  std::vector<T> predictor_scale_;
  std::vector<T> response_mean_;
  core::LdaModel<T> lda_;
  core::Matrix<T> direct_weights_;
  std::vector<T> direct_offsets_;
};

template<class T>
std::unique_ptr<ModelBase> fit_typed(
    const mxArray* x_value, const mxArray* y_value, const mxArray* options) {
  const auto x = mx_view<T>(x_value, "X");
  const Family family = parse_family(text_value(field(options, "method"), "Method"));
  const Head head = parse_head(text_value(field(options, "classifier"), "Classifier"));
  const auto scaling = parse_scaling(text_value(field(options, "scaling"), "Scaling"));
  const int requested = static_cast<int>(number_field(options, "components"));
  const int oversample = static_cast<int>(number_field(options, "oversample"));
  const int power = static_cast<int>(number_field(options, "power"));
  const int seed = static_cast<int>(number_field(options, "seed"));
  const int orthogonal = static_cast<int>(number_field(options, "orthogonalComponents"));
  const auto kernel_name = text_value(field(options, "kernel"), "Kernel");
  const double gamma = number_field(options, "gamma");
  const int degree = static_cast<int>(number_field(options, "degree"));
  const double offset = number_field(options, "offset");
  const bool store_scores = bool_field(options, "storeScores");
  if (x.rows() < 2 || x.columns() == 0 || requested < 1 || oversample < 0 ||
      power < 0 || seed < 0) {
    throw std::invalid_argument("invalid fitting dimensions or controls");
  }

  NativeBackend<T> backend;
  const std::size_t n = x.rows();
  const std::size_t p = x.columns();
  std::size_t classes = 0;
  std::vector<int> labels;
  core::Matrix<T> responses;
  if (head == Head::regression) {
    responses = copy_matrix(mx_view<T>(y_value, "Y"));
    if (responses.rows() != n) throw std::invalid_argument("X and Y rows differ");
  } else {
    if (mxGetClassID(y_value) != mxINT32_CLASS || mxIsComplex(y_value) ||
        mxGetNumberOfElements(y_value) != n) {
      throw std::invalid_argument("classification labels must be encoded int32 values");
    }
    const auto* encoded = static_cast<const std::int32_t*>(mxGetData(y_value));
    int maximum = -1;
    labels.resize(n);
    for (std::size_t row = 0; row < n; ++row) {
      if (encoded[row] < 0) throw std::invalid_argument("class indices must be non-negative");
      maximum = std::max(maximum, static_cast<int>(encoded[row]));
      labels[row] = static_cast<int>(encoded[row] + 1);
    }
    classes = static_cast<std::size_t>(maximum + 1);
    if (classes < 2) throw std::invalid_argument("classification requires two classes");
    if (family == Family::opls || family == Family::kernelpls) {
      responses.resize(n, classes);
      for (std::size_t row = 0; row < n; ++row) {
        responses(row, static_cast<std::size_t>(labels[row] - 1)) = T(1);
      }
    }
  }

  std::size_t cap = std::min(p, std::max<std::size_t>(n - 1, 1));
  if (family == Family::plssvd) {
    cap = std::min(cap, head == Head::regression ? responses.columns() :
                   std::max<std::size_t>(classes - 1, 1));
  }
  const std::size_t components = std::min<std::size_t>(requested, cap);
  auto simpls = simpls_controls(
      n, p, head == Head::regression ? responses.columns() : classes,
      components, head != Head::regression, oversample, power,
      static_cast<unsigned int>(seed));
  simpls.store_scores = store_scores ||
      (head == Head::lda &&
       (family == Family::opls || family == Family::kernelpls));

  std::vector<T> center;
  std::vector<T> scale;
  std::vector<T> response_mean;
  core::Matrix<T> class_predictor_sums;
  std::vector<T> class_counts;
  ModelVariant<T> fitted;
  if (family == Family::simpls || family == Family::plssvd) {
    if (head == Head::regression) {
      auto predictors = copy_matrix(x);
      const auto prepared = core::prepare_scaled_dense_crossprod(
          predictors.view(), responses.view(), scaling, backend);
      center = prepared.predictor_center;
      scale = prepared.predictor_scale;
      response_mean = prepared.response_mean;
      if (family == Family::simpls) {
        core::SimplsWorkspace<T> workspace;
        fitted = core::fit_simpls_preprocessed<T>(
            predictors.view(), prepared.crossprod.view(), simpls, backend, workspace);
      } else {
        const int count = static_cast<int>(components);
        core::PlssvdControls controls;
        controls.rsvd = simpls.rsvd;
        fitted = core::fit_plssvd_preprocessed<T>(
            predictors.view(), prepared.crossprod.view(), &count, 1, controls, backend);
      }
    } else {
      std::vector<std::size_t> zero_based(n);
      for (std::size_t row = 0; row < n; ++row) {
        zero_based[row] = static_cast<std::size_t>(labels[row] - 1);
      }
      const auto prepared = core::scaled_label_crossprod<T>(
          x, zero_based.data(), zero_based.size(), classes, scaling, backend);
      center = prepared.predictor_center;
      scale = prepared.predictor_scale;
      response_mean = prepared.response_mean;
      class_predictor_sums = prepared.class_predictor_sums;
      class_counts = prepared.class_counts;
      core::Matrix<T> predictor_gram(p, p);
      backend.self_gram(x, true, predictor_gram.view(), true);
      standardize_predictor_gram(
          predictor_gram.view(), n, center, scale, scaling);
      if (family == Family::simpls) {
        core::SimplsWorkspace<T> workspace;
        workspace.predictor_crossprod = std::move(predictor_gram);
        workspace.predictor_crossprod_preloaded = true;
        simpls.cache_predictor_crossprod = true;
        simpls.reorthogonalize = false;
        simpls.store_scores = false;
        simpls.store_score_moments = true;
        fitted = core::fit_simpls_preprocessed<T>(
            core::ConstMatrixView<T>(), prepared.crossprod.view(), simpls,
            backend, workspace, n);
      } else {
        const int count = static_cast<int>(components);
        core::PlssvdControls controls;
        controls.rsvd = simpls.rsvd;
        fitted = core::fit_plssvd_from_moments<T>(
            predictor_gram.view(), prepared.crossprod.view(), &count, 1,
            controls, backend);
      }
      if (store_scores) {
        auto standardized = copy_matrix(x);
        standardize(standardized, center, scale);
        auto& model_scores = family == Family::simpls ?
            std::get<core::SimplsModel<T>>(fitted).scores :
            std::get<core::PlssvdModel<T>>(fitted).scores;
        const auto& weights = family == Family::simpls ?
            std::get<core::SimplsModel<T>>(fitted).weights :
            std::get<core::PlssvdModel<T>>(fitted).weights;
        model_scores.resize(n, components);
        backend.gemm(standardized.view(), weights.view(), false, false,
                     model_scores.view());
      }
    }
  } else {
    auto predictors = copy_matrix(x);
    if (family == Family::opls) {
      core::OplsControls controls;
      controls.orthogonal_components = static_cast<std::size_t>(orthogonal);
      controls.scaling = scaling;
      controls.randomized_filter = true;
      controls.filter_rsvd = simpls.rsvd;
      controls.simpls = simpls;
      fitted = core::fit_opls(std::move(predictors), responses.view(), controls, backend);
      response_mean = std::get<core::OplsModel<T>>(fitted).response_mean;
    } else {
      core::KernelPlsControls controls;
      controls.kernel = parse_kernel(kernel_name);
      controls.gamma = gamma;
      controls.degree = degree;
      controls.offset = offset;
      controls.scaling = scaling;
      controls.simpls = simpls;
      fitted = core::fit_kernelpls(
          std::move(predictors), responses.view(), controls, backend);
      response_mean = std::get<core::KernelPlsModel<T>>(fitted).response_mean;
    }
  }

  core::LdaModel<T> lda;
  if (head == Head::lda) {
    const core::Matrix<T>* score_gram = nullptr;
    const core::Matrix<T>* score_matrix = nullptr;
    if (family == Family::simpls) {
      score_gram = &std::get<core::SimplsModel<T>>(fitted).score_gram;
      score_matrix = &std::get<core::SimplsModel<T>>(fitted).scores;
    } else if (family == Family::plssvd) {
      score_gram = &std::get<core::PlssvdModel<T>>(fitted).score_gram;
      score_matrix = &std::get<core::PlssvdModel<T>>(fitted).scores;
    } else if (family == Family::opls) {
      score_gram = &std::get<core::OplsModel<T>>(fitted).inner.score_gram;
      score_matrix = &std::get<core::OplsModel<T>>(fitted).inner.scores;
    } else {
      score_gram = &std::get<core::KernelPlsModel<T>>(fitted).inner.score_gram;
      score_matrix = &std::get<core::KernelPlsModel<T>>(fitted).inner.scores;
    }
    core::Matrix<T> class_sums(classes, components);
    std::vector<T> counts(classes, T(0));
    if (class_predictor_sums.size() != 0) {
      const auto& projection = family == Family::simpls ?
          std::get<core::SimplsModel<T>>(fitted).weights :
          std::get<core::PlssvdModel<T>>(fitted).weights;
      backend.gemm(class_predictor_sums.view(), projection.view(), true, false,
                   class_sums.view());
      counts = class_counts;
    } else {
      for (std::size_t row = 0; row < n; ++row) {
        const std::size_t label = static_cast<std::size_t>(labels[row] - 1);
        counts[label] += T(1);
        for (std::size_t component = 0; component < components; ++component) {
          class_sums(label, component) += (*score_matrix)(row, component);
        }
      }
    }
    const int count = static_cast<int>(components);
    auto models = core::train_lda_prefixes_from_moments<T>(
        score_gram->view(), class_sums.view(), counts.data(), counts.size(),
        n, &count, 1, backend);
    lda = std::move(models.front());
  }

  core::Matrix<T> direct_weights;
  std::vector<T> direct_offsets;
  if (head != Head::regression &&
      (family == Family::simpls || family == Family::plssvd)) {
    const auto& projection = family == Family::simpls ?
        std::get<core::SimplsModel<T>>(fitted).weights :
        std::get<core::PlssvdModel<T>>(fitted).weights;
    core::Matrix<T> latent(components, classes);
    if (head == Head::lda) {
      for (std::size_t category = 0; category < classes; ++category) {
        for (std::size_t component = 0; component < components; ++component) {
          latent(component, category) = lda.linear(category, component);
        }
      }
      direct_offsets = lda.constants;
    } else if (family == Family::simpls) {
      const auto& loadings = std::get<core::SimplsModel<T>>(fitted).response_loadings;
      for (std::size_t category = 0; category < classes; ++category) {
        for (std::size_t component = 0; component < components; ++component) {
          latent(component, category) = loadings(category, component);
        }
      }
      direct_offsets = response_mean;
    } else {
      latent = std::get<core::PlssvdModel<T>>(fitted).prediction_weights.front();
      direct_offsets = response_mean;
    }
    direct_weights.resize(p, classes);
    backend.gemm(projection.view(), latent.view(), false, false, direct_weights.view());
    for (std::size_t predictor = 0; predictor < p; ++predictor) {
      const T inverse = T(1) / scale[predictor];
      const T centered = center[predictor] * inverse;
      for (std::size_t category = 0; category < classes; ++category) {
        const T original = direct_weights(predictor, category);
        direct_offsets[category] -= centered * original;
        direct_weights(predictor, category) = original * inverse;
      }
    }
  }

  return std::make_unique<Model<T>>(
      family, head, components, classes, std::move(fitted), std::move(center),
      std::move(scale), std::move(response_mean), std::move(lda),
      std::move(direct_weights), std::move(direct_offsets));
}

std::unordered_set<ModelBase*> models;

void cleanup() {
  for (ModelBase* model : models) delete model;
  models.clear();
}

ModelBase* handle_value(const mxArray* value) {
  if (value == nullptr || mxGetClassID(value) != mxUINT64_CLASS ||
      mxGetNumberOfElements(value) != 1) {
    throw std::invalid_argument("invalid native model handle");
  }
  const auto raw = *static_cast<const std::uint64_t*>(mxGetData(value));
  auto* model = reinterpret_cast<ModelBase*>(raw);
  if (models.find(model) == models.end()) {
    throw std::invalid_argument("native model handle is stale or invalid");
  }
  return model;
}

mxArray* create_handle(ModelBase* model) {
  mxArray* output = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<std::uint64_t*>(mxGetData(output)) =
      reinterpret_cast<std::uint64_t>(model);
  return output;
}

template<class T>
void randomized_svd_outputs(const mxArray* input, int components,
                            int oversample, int power, unsigned int seed,
                            mxArray* output[]) {
  const auto values = mx_view<T>(input, "X");
  if (components < 1 || static_cast<std::size_t>(components) >
      std::min(values.rows(), values.columns())) {
    throw std::invalid_argument("NumComponents exceeds the matrix rank bound");
  }
  core::RsvdControls controls;
  controls.oversample = oversample;
  controls.power = power;
  controls.seed = seed;
  controls.left_only = false;
  NativeBackend<T> backend;
  auto decomposition = core::randomized_svd(
      values, components, controls, backend);
  output[0] = mx_matrix(decomposition.U);
  const mxClassID type = std::is_same<T, float>::value ?
      mxSINGLE_CLASS : mxDOUBLE_CLASS;
  output[1] = mxCreateNumericMatrix(
      static_cast<mwSize>(decomposition.singular_values.size()), 1, type, mxREAL);
  std::copy(decomposition.singular_values.begin(),
            decomposition.singular_values.end(),
            static_cast<T*>(mxGetData(output[1])));
  core::Matrix<T> right(decomposition.Vt.columns(), decomposition.Vt.rows());
  for (std::size_t column = 0; column < right.columns(); ++column) {
    for (std::size_t row = 0; row < right.rows(); ++row) {
      right(row, column) = decomposition.Vt(column, row);
    }
  }
  output[2] = mx_matrix(right);
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  try {
    if (nrhs < 1) throw std::invalid_argument("a command is required");
    const std::string command = text_value(prhs[0], "command");
    if (command == "fastsvd") {
      if (nrhs != 6 || nlhs != 3) {
        throw std::invalid_argument(
            "fastsvd requires X, components, oversample, power, and seed");
      }
      const int components = static_cast<int>(mxGetScalar(prhs[2]));
      const int oversample = static_cast<int>(mxGetScalar(prhs[3]));
      const int power = static_cast<int>(mxGetScalar(prhs[4]));
      const unsigned int seed = static_cast<unsigned int>(mxGetScalar(prhs[5]));
      if (mxGetClassID(prhs[1]) == mxSINGLE_CLASS) {
        randomized_svd_outputs<float>(
            prhs[1], components, oversample, power, seed, plhs);
      } else if (mxGetClassID(prhs[1]) == mxDOUBLE_CLASS) {
        randomized_svd_outputs<double>(
            prhs[1], components, oversample, power, seed, plhs);
      } else {
        throw std::invalid_argument("X must be single or double");
      }
      return;
    }
    if (command == "new") {
      if (nrhs != 4 || nlhs < 1) {
        throw std::invalid_argument("new requires X, Y, and options");
      }
      std::unique_ptr<ModelBase> model;
      if (mxGetClassID(prhs[1]) == mxSINGLE_CLASS) {
        model = fit_typed<float>(prhs[1], prhs[2], prhs[3]);
      } else if (mxGetClassID(prhs[1]) == mxDOUBLE_CLASS) {
        model = fit_typed<double>(prhs[1], prhs[2], prhs[3]);
      } else {
        throw std::invalid_argument("X must be single or double");
      }
      ModelBase* raw = model.release();
      models.insert(raw);
      mexLock();
      static bool registered = false;
      if (!registered) {
        mexAtExit(cleanup);
        registered = true;
      }
      plhs[0] = create_handle(raw);
      if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(raw->components());
      return;
    }
    if (command == "delete") {
      if (nrhs != 2) throw std::invalid_argument("delete requires a handle");
      ModelBase* model = handle_value(prhs[1]);
      models.erase(model);
      delete model;
      mexUnlock();
      return;
    }
    if (command == "predict" || command == "scores" || command == "classes") {
      if (nrhs < 3 || nlhs != 1) {
        throw std::invalid_argument("prediction requires a handle and X");
      }
      ModelBase* model = handle_value(prhs[1]);
      const bool input_single = mxGetClassID(prhs[2]) == mxSINGLE_CLASS;
      if (input_single != model->is_single()) {
        throw std::invalid_argument("prediction precision must match training precision");
      }
      if (command == "predict") plhs[0] = model->predict(prhs[2]);
      if (command == "scores") plhs[0] = model->scores(prhs[2]);
      if (command == "classes") {
        if (nrhs != 4) throw std::invalid_argument("classes requires Top");
        plhs[0] = model->classes(
            prhs[2], static_cast<std::size_t>(mxGetScalar(prhs[3])));
      }
      return;
    }
    if (command == "backend") {
      plhs[0] = mxCreateString(fastpls_matlab::backend_info().c_str());
      return;
    }
    throw std::invalid_argument("unknown fastPLS MEX command");
  } catch (const std::exception& error) {
    mexErrMsgIdAndTxt("fastPLS:NativeError", "%s", error.what());
  }
}
