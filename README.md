<p align="center">
  <img src="assets/logo.png" alt="tinycml logo" width="300">
</p>

<h1 align="center">tinycml</h1>

<p align="center">
  <strong>A High-Performance Machine Learning Library in Pure C</strong>
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

# English

## Abstract

**tinycml** is a comprehensive machine learning library implemented entirely in the C programming language (ISO C11). The library provides a complete suite of supervised and unsupervised learning algorithms, preprocessing utilities, model selection infrastructure, and evaluation metrics—all without external dependencies beyond the C standard library.

The implementation prioritizes algorithmic transparency, computational efficiency, and deployment flexibility across heterogeneous computing environments, from high-performance servers to resource-constrained embedded systems.

| Metric | Value |
|--------|-------|
| **Total Source Lines** | 11,600+ |
| **Test Coverage** | 127 tests / 30 suites |
| **Compiled Binary Size** | 210 KB (static), 220 KB (shared) |
| **External Dependencies** | 0 |
| **Language Standard** | ISO C11 |
| **Cold Start Latency** | < 1 ms |

## Comparative Analysis

### Runtime Performance Characteristics

The following table presents a comparative analysis of tinycml against prevalent machine learning frameworks. Measurements reflect typical deployment scenarios on contemporary hardware.

| Metric | tinycml | scikit-learn | TensorFlow | PyTorch |
|--------|---------|--------------|------------|---------|
| **Initialization Latency** | < 1 ms | 800–1200 ms | 2000–4000 ms | 1500–3000 ms |
| **Memory Footprint (Idle)** | 160 KB | 150–300 MB | 400–800 MB | 300–600 MB |
| **Dependency Count** | 0 | 12+ (NumPy, SciPy, joblib, etc.) | 50+ | 30+ |
| **Python Interpreter Required** | No | Yes | Yes | Yes |
| **Virtual Environment Required** | No | Recommended | Required | Required |
| **GIL Contention** | N/A | Yes | Partial | Partial |

### Deployment Flexibility

| Capability | tinycml | scikit-learn | TensorFlow | PyTorch |
|------------|---------|--------------|------------|---------|
| **Bare-Metal Execution** | Yes | No | No | No |
| **Microcontroller Deployment** | Yes | No | TFLite only | No |
| **Static Linking** | Yes | No | Limited | No |
| **Cross-Compilation** | Trivial | Complex | Complex | Complex |
| **Air-Gapped Installation** | Single file | Package ecosystem | Package ecosystem | Package ecosystem |
| **Deterministic Execution** | Yes | No (GC, JIT) | No | No |

### Algorithmic Transparency

| Aspect | tinycml | scikit-learn | TensorFlow | PyTorch |
|--------|---------|--------------|------------|---------|
| **Lines per Algorithm** | 100–400 | 500–2000+ | 1000–10000+ | 1000–5000+ |
| **Abstraction Layers** | 1–2 | 3–5 | 5–10+ | 4–8+ |
| **Step-Through Debugging** | Native GDB/LLDB | Python debugger | Complex | Complex |
| **Algorithm Modification** | Direct source edit | Subclassing required | Custom ops required | Custom modules required |

### Use Case Recommendations

The following tables provide objective guidance for framework selection based on specific requirements and constraints.

#### When to Choose tinycml

| Use Case | Rationale |
|----------|-----------|
| **Embedded Systems & IoT** | 160KB footprint fits microcontrollers; no OS dependencies required |
| **Real-Time Applications** | Deterministic execution without GC pauses; sub-millisecond latency |
| **Educational Purposes** | Readable algorithms (100–400 lines each); step-through debugging with GDB |
| **Air-Gapped Environments** | Single-file deployment; no package manager or network access needed |
| **Resource-Constrained Devices** | Minimal RAM usage; static memory allocation patterns |
| **Safety-Critical Systems** | Auditable codebase; deterministic behavior; no dynamic dependencies |
| **Cross-Platform Deployment** | Compile once for any target with a C compiler |
| **Algorithm Research & Modification** | Direct source access; no abstraction layers to navigate |

#### When to Choose Python Frameworks (scikit-learn, TensorFlow, PyTorch)

| Use Case | Rationale |
|----------|-----------|
| **Rapid Prototyping** | Interactive development; immediate visualization; Jupyter integration |
| **GPU Acceleration** | CUDA/cuDNN support for large-scale neural network training |
| **Deep Learning at Scale** | Optimized tensor operations; distributed training; TPU support |
| **Large Dataset Processing** | Out-of-core learning; Dask/Spark integration; memory-mapped arrays |
| **Pre-trained Models** | Extensive model zoos; transfer learning; fine-tuning capabilities |
| **Production ML Pipelines** | MLflow, Kubeflow integration; model versioning; A/B testing infrastructure |
| **Computer Vision & NLP** | Specialized architectures (CNNs, Transformers); pre-processing pipelines |
| **AutoML & Hyperparameter Optimization** | Optuna, Ray Tune integration; neural architecture search |
| **Team Collaboration** | Familiar Python ecosystem; extensive documentation; large community |

#### Decision Matrix

| Requirement | Recommended Framework |
|-------------|----------------------|
| Binary size < 1 MB | tinycml |
| No Python runtime available | tinycml |
| GPU training required | TensorFlow / PyTorch |
| Microcontroller deployment | tinycml |
| Deep neural networks (10+ layers) | TensorFlow / PyTorch |
| Algorithm transparency for teaching | tinycml |
| Pre-trained model fine-tuning | TensorFlow / PyTorch |
| Deterministic, reproducible execution | tinycml |
| Rapid experimentation with visualization | scikit-learn / PyTorch |
| Safety-critical certification requirements | tinycml |
| Large-scale distributed training | TensorFlow / PyTorch |
| Minimal deployment dependencies | tinycml |

## Theoretical Foundation

### Computational Model

tinycml operates within a deterministic computational model characterized by:

1. **Explicit Memory Management**: All allocations and deallocations occur at precisely defined points in the execution flow, enabling accurate memory budgeting for embedded deployments.

2. **Absence of Runtime Overhead**: No interpreter, just-in-time compiler, or garbage collector introduces latency variance. Function invocation proceeds directly to native machine instructions.

3. **Cache-Coherent Data Structures**: Matrix representations utilize contiguous row-major storage, optimizing spatial locality for modern CPU cache hierarchies.

4. **Portable Numeric Semantics**: IEEE 754 double-precision arithmetic ensures reproducible results across conforming platforms.

### Architectural Design

The library employs a unified estimator interface inspired by established machine learning API conventions:

```c
typedef struct Estimator {
    ModelType type;
    TaskType task;
    int is_fitted;

    // Polymorphic interface via function pointers
    struct Estimator* (*fit)(struct Estimator*, const Matrix*, const Matrix*);
    Matrix* (*predict)(const struct Estimator*, const Matrix*);
    double (*score)(const struct Estimator*, const Matrix*, const Matrix*);
    Matrix* (*transform)(struct Estimator*, const Matrix*);
    void (*free)(struct Estimator*);
} Estimator;
```

This design enables algorithm-agnostic pipelines and cross-validation infrastructure while preserving C's performance characteristics.

## Implemented Algorithms

### Supervised Learning

| Algorithm | Methodology | Complexity |
|-----------|-------------|------------|
| **Linear Regression** | Closed-form solution via normal equations; iterative gradient descent with configurable learning rate | O(n·p²) closed-form; O(n·p·k) iterative |
| **Logistic Regression** | Maximum likelihood estimation with L2 regularization; gradient descent optimization | O(n·p·k) |
| **k-Nearest Neighbors** | Brute-force distance computation; Euclidean metric; distance-weighted voting | O(n·m·p) prediction |
|| **Naive Bayes** | Gaussian likelihood estimation; maximum a posteriori classification | O(n·p) training; O(m·p·c) prediction |
| **Decision Tree** | Recursive partitioning; Gini impurity and information gain criteria; configurable depth constraints | O(n·p·log n) |
| **Random Forest** | Bootstrap aggregation; random feature subspace selection; out-of-bag error estimation | O(t·n·p·log n) |
| **Support Vector Machine** | Linear and RBF kernels; hinge loss minimization; sub-gradient descent; Platt scaling for probability estimates | O(n·p·k) |
| **Softmax Regression** | Multinomial logistic regression; cross-entropy loss; gradient descent | O(n·p·c·k) |
| **Ridge Regression** | L2-regularized linear regression; closed-form via augmented normal equations | O(n·p² + p³) |
| **Lasso Regression** | L1-regularized linear regression; coordinate descent; automatic feature selection | O(n·p·k) |
| **Neural Network** | Fully-connected feedforward architecture; backpropagation; mini-batch stochastic gradient descent | O(n·Σ(lᵢ·lᵢ₊₁)·k) |

### Unsupervised Learning

| Algorithm | Methodology | Complexity |
|-----------|-------------|------------|
| **k-Means Clustering** | Lloyd's algorithm with k-means++ initialization; iterative centroid refinement | O(n·k·p·i) |
| **DBSCAN** | Density-based spatial clustering; core/border/noise classification; configurable epsilon and min_samples | O(n²) |
| **Principal Component Analysis** | Covariance matrix eigendecomposition; variance-maximizing projection; optional whitening transformation | O(n·p² + p³) |

### Feature Engineering

| Component | Function |
|-----------|----------|
| **SelectKBest** | Univariate feature selection via F-statistic, χ² test, or mutual information |
| **VarianceThreshold** | Elimination of quasi-constant features below specified variance threshold |
| **StandardScaler** | Z-score normalization: zero mean, unit variance transformation |
| **MinMaxScaler** | Linear scaling to specified range [a, b] |
|| **OneHotEncoder** | Categorical variable expansion to binary indicator vectors |
| **PolynomialFeatures** | Generation of polynomial and interaction terms up to specified degree |

### Model Selection Infrastructure

| Component | Function |
|-----------|----------|
| **Cross-Validation** | K-fold and stratified K-fold partitioning with aggregated performance metrics |
| **GridSearchCV** | Exhaustive hyperparameter search with cross-validated evaluation |
| **Pipeline** | Sequential composition of transformers and estimators |
| **Model Serialization** | Binary persistence with magic header for Linear/Logistic Regression, SVM, Gaussian/Multinomial Naive Bayes |

### Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Regression** | Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, R² Coefficient of Determination |
| **Classification** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Clustering** | Inertia (within-cluster sum of squares), Silhouette Coefficient |

## System Requirements

### Compiler Compatibility

| Compiler | Minimum Version |
|----------|-----------------|
| GCC | 4.7 |
| Clang | 3.1 |
| MSVC | 2015 |

### Target Platforms

The library has been validated on the following platforms:

- Linux (x86_64, ARM64, ARMv7)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64)
- FreeBSD, OpenBSD
- Embedded Linux (Raspberry Pi, BeagleBone)
- Bare-metal ARM Cortex-M (with appropriate libc)

## Compilation and Installation

```bash
# Clone repository
git clone https://github.com/tinylabs/tinycml.git
cd tinycml

# Build library and examples
make

# Execute test suite
make test

# Build static library only
make library
```

## Usage Examples

### Fundamental Pattern

All estimators adhere to a consistent interface:

```c
#include "linear_regression.h"

// Instantiation
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Parameter estimation
model->base.fit((Estimator*)model, X_train, y_train);

// Inference
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Performance evaluation
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Resource deallocation
model->base.free((Estimator*)model);
```

### Pipeline Construction

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "logistic_regression.h"

Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "classifier", (Estimator*)logistic_regression_create());

pipe->base.fit((Estimator*)pipe, X_train, y_train);
Matrix *predictions = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

### Neural Network Configuration

```c
#include "neural_network.h"

size_t architecture[] = {784, 256, 128, 10};
NeuralNetwork *nn = neural_network_create(architecture, 4, ACTIVATION_RELU);

nn->learning_rate = 0.001;
nn->epochs = 100;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);
double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);

nn->base.free((Estimator*)nn);
```

### Ensemble Methods

```c
#include "ensemble.h"

RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,    // n_estimators
    15,     // max_depth
    2,      // min_samples_split
    1,      // min_samples_leaf
    0,      // max_features (0 = sqrt(n_features))
    1,      // bootstrap
    42      // random_state
);

rf->base.fit((Estimator*)rf, X_train, y_train);
printf("Out-of-Bag Score: %.4f\n", rf->oob_score_);

rf->base.free((Estimator*)rf);
```

### Cross-Validation

```c
#include "validation.h"

DecisionTreeClassifier *dt = decision_tree_classifier_create();
CrossValResults *cv = cross_val_score((Estimator*)dt, X, y, 5, 1, 42);

printf("Mean Accuracy: %.4f (±%.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
dt->base.free((Estimator*)dt);
```

### Hyperparameter Optimization

```c
#include "model_selection.h"

ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){5, 10, 15, 20}, 4);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

gs->base.fit((Estimator*)gs, X, y);
printf("Optimal Score: %.4f\n", gs->best_score_);

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### Dimensionality Reduction

```c
#include "decomposition.h"

PCA *pca = pca_create(2);
pca->base.fit((Estimator*)pca, X, NULL);

Matrix *X_projected = pca->base.transform((Estimator*)pca, X);

const double *explained_variance = pca_explained_variance_ratio(pca);
printf("Cumulative Explained Variance: %.2f%%\n",
       (explained_variance[0] + explained_variance[1]) * 100);

pca->base.free((Estimator*)pca);
```

### Feature Selection

```c
#include "feature_selection.h"

// Statistical feature ranking
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 10);
selector->base.fit((Estimator*)selector, X, y);
Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// Variance-based filtering
VarianceThreshold *vt = variance_threshold_create(0.01);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

## Project Architecture

```
tinycml/
├── include/                # Public header files
│   ├── matrix.h            # Matrix and vector operations
│   ├── estimator.h         # Base estimator interface
│   ├── pipeline.h          # Pipeline infrastructure
│   ├── validation.h        # Cross-validation utilities
│   ├── model_selection.h   # GridSearchCV implementation
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── svm.h
│   ├── dbscan.h
│   ├── ridge.h
│   ├── lasso.h
│   ├── cml_error.h        # Unified error system
│   ├── cml_serialization.h # Model save/load
│   ├── ensemble.h          # Random Forest
│   ├── neural_network.h
│   ├── decomposition.h     # PCA
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/                    # Implementation files
├── examples/               # Executable demonstrations
├── tests/                  # Unit test suite
├── data/                   # Sample datasets
└── docs/                   # API documentation
```

## License

This software is distributed under the MIT License.

---

# Türkçe

## Özet

**tinycml**, tamamen C programlama dilinde (ISO C11) geliştirilmiş kapsamlı bir makine öğrenmesi kütüphanesidir. Kütüphane; denetimli ve denetimsiz öğrenme algoritmaları, ön işleme araçları, model seçim altyapısı ve değerlendirme metriklerinden oluşan eksiksiz bir set sunar—tamamı C standart kütüphanesi dışında herhangi bir harici bağımlılık olmaksızın.

Uygulama; algoritmik şeffaflık, hesaplama verimliliği ve yüksek performanslı sunuculardan kaynak kısıtlı gömülü sistemlere kadar heterojen bilgi işlem ortamlarında dağıtım esnekliğini ön planda tutar.

| Metrik | Değer |
|--------|-------|
|| **Toplam Kaynak Satırı** | 11.600+ |
|| **Test Kapsamı** | 127 test / 30 suit |
|| **Derlenmiş Binary Boyutu** | 210 KB (statik), 220 KB (paylaşımlı) |
| **Harici Bağımlılık** | 0 |
| **Dil Standardı** | ISO C11 |
| **Soğuk Başlangıç Gecikmesi** | < 1 ms |

## Karşılaştırmalı Analiz

### Çalışma Zamanı Performans Özellikleri

Aşağıdaki tablo, tinycml'in yaygın makine öğrenmesi çerçeveleriyle karşılaştırmalı analizini sunmaktadır. Ölçümler, güncel donanım üzerindeki tipik dağıtım senaryolarını yansıtmaktadır.

| Metrik | tinycml | scikit-learn | TensorFlow | PyTorch |
|--------|---------|--------------|------------|---------|
| **Başlatma Gecikmesi** | < 1 ms | 800–1200 ms | 2000–4000 ms | 1500–3000 ms |
| **Bellek Ayak İzi (Boşta)** | 160 KB | 150–300 MB | 400–800 MB | 300–600 MB |
| **Bağımlılık Sayısı** | 0 | 12+ (NumPy, SciPy, joblib, vb.) | 50+ | 30+ |
| **Python Yorumlayıcısı Gerekli** | Hayır | Evet | Evet | Evet |
| **Sanal Ortam Gerekli** | Hayır | Önerilir | Zorunlu | Zorunlu |
| **GIL Çekişmesi** | Yok | Evet | Kısmi | Kısmi |

### Dağıtım Esnekliği

| Yetenek | tinycml | scikit-learn | TensorFlow | PyTorch |
|---------|---------|--------------|------------|---------|
| **Bare-Metal Çalıştırma** | Evet | Hayır | Hayır | Hayır |
| **Mikrodenetleyici Dağıtımı** | Evet | Hayır | Yalnızca TFLite | Hayır |
| **Statik Bağlama** | Evet | Hayır | Sınırlı | Hayır |
| **Çapraz Derleme** | Basit | Karmaşık | Karmaşık | Karmaşık |
| **Ağ Bağlantısız Kurulum** | Tek dosya | Paket ekosistemi | Paket ekosistemi | Paket ekosistemi |
| **Deterministik Çalıştırma** | Evet | Hayır (GC, JIT) | Hayır | Hayır |

### Algoritmik Şeffaflık

| Husus | tinycml | scikit-learn | TensorFlow | PyTorch |
|-------|---------|--------------|------------|---------|
| **Algoritma Başına Satır** | 100–400 | 500–2000+ | 1000–10000+ | 1000–5000+ |
| **Soyutlama Katmanları** | 1–2 | 3–5 | 5–10+ | 4–8+ |
| **Adım Adım Hata Ayıklama** | Native GDB/LLDB | Python hata ayıklayıcı | Karmaşık | Karmaşık |
| **Algoritma Modifikasyonu** | Doğrudan kaynak düzenleme | Alt sınıflama gerekli | Özel operatörler gerekli | Özel modüller gerekli |

### Kullanım Senaryosu Önerileri

Aşağıdaki tablolar, belirli gereksinimler ve kısıtlamalar temelinde çerçeve seçimi için objektif rehberlik sağlar.

#### tinycml Ne Zaman Tercih Edilmeli

| Kullanım Senaryosu | Gerekçe |
|--------------------|---------|
| **Gömülü Sistemler ve IoT** | 160KB ayak izi mikrodenetleyicilere sığar; işletim sistemi bağımlılığı gerekmez |
| **Gerçek Zamanlı Uygulamalar** | GC duraklamaları olmadan deterministik çalıştırma; milisaniye altı gecikme |
| **Eğitim Amaçlı** | Okunabilir algoritmalar (her biri 100–400 satır); GDB ile adım adım hata ayıklama |
| **Ağ Bağlantısız Ortamlar** | Tek dosya dağıtımı; paket yöneticisi veya ağ erişimi gerekmez |
| **Kaynak Kısıtlı Cihazlar** | Minimal RAM kullanımı; statik bellek tahsis kalıpları |
| **Güvenlik Kritik Sistemler** | Denetlenebilir kod tabanı; deterministik davranış; dinamik bağımlılık yok |
| **Çapraz Platform Dağıtımı** | C derleyicisi olan herhangi bir hedef için bir kez derle |
| **Algoritma Araştırma ve Modifikasyonu** | Doğrudan kaynak erişimi; gezinilecek soyutlama katmanı yok |

#### Python Çerçeveleri Ne Zaman Tercih Edilmeli (scikit-learn, TensorFlow, PyTorch)

| Kullanım Senaryosu | Gerekçe |
|--------------------|---------|
| **Hızlı Prototipleme** | İnteraktif geliştirme; anlık görselleştirme; Jupyter entegrasyonu |
| **GPU Hızlandırma** | Büyük ölçekli sinir ağı eğitimi için CUDA/cuDNN desteği |
| **Ölçekte Derin Öğrenme** | Optimize tensör işlemleri; dağıtık eğitim; TPU desteği |
| **Büyük Veri Seti İşleme** | Bellek dışı öğrenme; Dask/Spark entegrasyonu; bellek eşlemeli diziler |
| **Önceden Eğitilmiş Modeller** | Kapsamlı model havuzları; transfer öğrenme; ince ayar yetenekleri |
| **Üretim ML Pipeline'ları** | MLflow, Kubeflow entegrasyonu; model versiyonlama; A/B test altyapısı |
| **Bilgisayarlı Görü ve NLP** | Özelleşmiş mimariler (CNN'ler, Transformer'lar); ön işleme pipeline'ları |
| **AutoML ve Hiperparametre Optimizasyonu** | Optuna, Ray Tune entegrasyonu; sinir mimarisi araması |
| **Takım İşbirliği** | Tanıdık Python ekosistemi; kapsamlı dokümantasyon; geniş topluluk |

#### Karar Matrisi

| Gereksinim | Önerilen Çerçeve |
|------------|------------------|
| Binary boyutu < 1 MB | tinycml |
| Python çalışma zamanı mevcut değil | tinycml |
| GPU eğitimi gerekli | TensorFlow / PyTorch |
| Mikrodenetleyici dağıtımı | tinycml |
| Derin sinir ağları (10+ katman) | TensorFlow / PyTorch |
| Eğitim için algoritma şeffaflığı | tinycml |
| Önceden eğitilmiş model ince ayarı | TensorFlow / PyTorch |
| Deterministik, tekrarlanabilir çalıştırma | tinycml |
| Görselleştirme ile hızlı deneyler | scikit-learn / PyTorch |
| Güvenlik kritik sertifikasyon gereksinimleri | tinycml |
| Büyük ölçekli dağıtık eğitim | TensorFlow / PyTorch |
| Minimal dağıtım bağımlılıkları | tinycml |

## Teorik Temel

### Hesaplama Modeli

tinycml, aşağıdaki özelliklerle karakterize edilen deterministik bir hesaplama modeli içinde çalışır:

1. **Açık Bellek Yönetimi**: Tüm tahsisler ve serbest bırakmalar, yürütme akışında tam olarak tanımlanmış noktalarda gerçekleşir, bu da gömülü dağıtımlar için doğru bellek bütçelemesini mümkün kılar.

2. **Çalışma Zamanı Yükü Yokluğu**: Hiçbir yorumlayıcı, tam zamanında derleyici veya çöp toplayıcı gecikme varyansı oluşturmaz. Fonksiyon çağrısı doğrudan yerel makine talimatlarına ilerler.

3. **Önbellek Tutarlı Veri Yapıları**: Matris gösterimleri bitişik satır-öncelikli depolama kullanır, modern CPU önbellek hiyerarşileri için uzamsal yerelliği optimize eder.

4. **Taşınabilir Sayısal Semantik**: IEEE 754 çift duyarlıklı aritmetik, uyumlu platformlarda tekrarlanabilir sonuçlar sağlar.

### Mimari Tasarım

Kütüphane, yerleşik makine öğrenmesi API geleneklerinden esinlenen birleşik bir tahmin edici arayüzü kullanır:

```c
typedef struct Estimator {
    ModelType type;
    TaskType task;
    int is_fitted;

    // Fonksiyon işaretçileri ile polimorfik arayüz
    struct Estimator* (*fit)(struct Estimator*, const Matrix*, const Matrix*);
    Matrix* (*predict)(const struct Estimator*, const Matrix*);
    double (*score)(const struct Estimator*, const Matrix*, const Matrix*);
    Matrix* (*transform)(struct Estimator*, const Matrix*);
    void (*free)(struct Estimator*);
} Estimator;
```

Bu tasarım, C'nin performans özelliklerini korurken algoritmadan bağımsız pipeline'lar ve çapraz doğrulama altyapısını mümkün kılar.

## Uygulanan Algoritmalar

### Denetimli Öğrenme

| Algoritma | Metodoloji | Karmaşıklık |
|-----------|------------|-------------|
| **Lineer Regresyon** | Normal denklemler aracılığıyla kapalı form çözümü; yapılandırılabilir öğrenme oranı ile yinelemeli gradyan inişi | O(n·p²) kapalı form; O(n·p·k) yinelemeli |
| **Lojistik Regresyon** | L2 düzenlileştirme ile maksimum olabilirlik tahmini; gradyan iniş optimizasyonu | O(n·p·k) |
| **k-En Yakın Komşu** | Kaba kuvvet mesafe hesaplaması; Öklid metriği; mesafe ağırlıklı oylama | O(n·m·p) tahmin |
|| **Naive Bayes** | Gaussian olabilirlik tahmini; maksimum sonsal sınıflandırma | O(n·p) eğitim; O(m·p·c) tahmin |
| **Karar Ağacı** | Özyinelemeli bölümleme; Gini safsızlığı ve bilgi kazancı kriterleri; yapılandırılabilir derinlik kısıtlamaları | O(n·p·log n) |
| **Rastgele Orman** | Bootstrap toplama; rastgele özellik alt uzayı seçimi; torba dışı hata tahmini | O(t·n·p·log n) |
|| **Destek Vektör Makinesi** | Lineer çekirdek; menteşe kaybı minimizasyonu; alt-gradyan inişi | O(n·p·k) |
| **Sinir Ağı** | Tam bağlantılı ileri beslemeli mimari; geri yayılım; mini-batch stokastik gradyan inişi | O(n·Σ(lᵢ·lᵢ₊₁)·k) |

### Denetimsiz Öğrenme

| Algoritma | Metodoloji | Karmaşıklık |
|-----------|------------|-------------|
| **k-Means Kümeleme** | k-means++ başlatma ile Lloyd algoritması; yinelemeli merkez iyileştirme | O(n·k·p·i) |
| **Temel Bileşen Analizi** | Kovaryans matrisi özdeğer ayrışımı; varyans maksimize eden projeksiyon; isteğe bağlı beyazlatma dönüşümü | O(n·p² + p³) |

### Özellik Mühendisliği

| Bileşen | İşlev |
|---------|-------|
| **SelectKBest** | F-istatistiği, χ² testi veya karşılıklı bilgi ile tek değişkenli özellik seçimi |
| **VarianceThreshold** | Belirtilen varyans eşiğinin altındaki yarı sabit özelliklerin elenmesi |
| **StandardScaler** | Z-skoru normalizasyonu: sıfır ortalama, birim varyans dönüşümü |
| **MinMaxScaler** | Belirtilen [a, b] aralığına lineer ölçekleme |
|| **OneHotEncoder** | Kategorik değişkenlerin ikili gösterge vektörlerine genişletilmesi |
| **PolynomialFeatures** | Belirtilen dereceye kadar polinom ve etkileşim terimlerinin üretimi |

### Model Seçim Altyapısı

| Bileşen | İşlev |
|---------|-------|
| **Çapraz Doğrulama** | Toplu performans metrikleri ile K-katlı ve tabakalı K-katlı bölümleme |
| **GridSearchCV** | Çapraz doğrulamalı değerlendirme ile kapsamlı hiperparametre araması |
| **Pipeline** | Dönüştürücüler ve tahmin edicilerin sıralı kompozisyonu |
| **Model Serileştirme** | Eğitilmiş model parametrelerinin ikili kalıcılığı |

### Değerlendirme Metrikleri

| Kategori | Metrikler |
|----------|-----------|
| **Regresyon** | Ortalama Kare Hata, Kök Ortalama Kare Hata, Ortalama Mutlak Hata, R² Belirleme Katsayısı |
| **Sınıflandırma** | Doğruluk, Kesinlik, Duyarlılık, F1-Skoru, Karışıklık Matrisi |
| **Kümeleme** | Atalet (küme içi kareler toplamı), Silhouette Katsayısı |

## Sistem Gereksinimleri

### Derleyici Uyumluluğu

| Derleyici | Minimum Sürüm |
|-----------|---------------|
| GCC | 4.7 |
| Clang | 3.1 |
| MSVC | 2015 |

### Hedef Platformlar

Kütüphane aşağıdaki platformlarda doğrulanmıştır:

- Linux (x86_64, ARM64, ARMv7)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64)
- FreeBSD, OpenBSD
- Gömülü Linux (Raspberry Pi, BeagleBone)
- Bare-metal ARM Cortex-M (uygun libc ile)

## Derleme ve Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/tinylabs/tinycml.git
cd tinycml

# Kütüphane ve örnekleri derleyin
make

# Test paketini çalıştırın
make test

# Yalnızca statik kütüphaneyi derleyin
make library
```

## Kullanım Örnekleri

### Temel Desen

Tüm tahmin ediciler tutarlı bir arayüze uyar:

```c
#include "linear_regression.h"

// Örnekleme
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Parametre tahmini
model->base.fit((Estimator*)model, X_train, y_train);

// Çıkarım
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Performans değerlendirmesi
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Kaynak serbest bırakma
model->base.free((Estimator*)model);
```

### Pipeline Oluşturma

```c
#include "pipeline.h"
#include "preprocessing.h"
#include "logistic_regression.h"

Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "classifier", (Estimator*)logistic_regression_create());

pipe->base.fit((Estimator*)pipe, X_train, y_train);
Matrix *predictions = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

### Sinir Ağı Yapılandırması

```c
#include "neural_network.h"

size_t architecture[] = {784, 256, 128, 10};
NeuralNetwork *nn = neural_network_create(architecture, 4, ACTIVATION_RELU);

nn->learning_rate = 0.001;
nn->epochs = 100;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);
double accuracy = nn->base.score((Estimator*)nn, X_test, y_test);

nn->base.free((Estimator*)nn);
```

### Topluluk Yöntemleri

```c
#include "ensemble.h"

RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,    // n_estimators
    15,     // max_depth
    2,      // min_samples_split
    1,      // min_samples_leaf
    0,      // max_features (0 = sqrt(n_features))
    1,      // bootstrap
    42      // random_state
);

rf->base.fit((Estimator*)rf, X_train, y_train);
printf("Torba Dışı Skor: %.4f\n", rf->oob_score_);

rf->base.free((Estimator*)rf);
```

### Çapraz Doğrulama

```c
#include "validation.h"

DecisionTreeClassifier *dt = decision_tree_classifier_create();
CrossValResults *cv = cross_val_score((Estimator*)dt, X, y, 5, 1, 42);

printf("Ortalama Doğruluk: %.4f (±%.4f)\n", cv->mean_test_score, cv->std_test_score);

cross_val_results_free(cv);
dt->base.free((Estimator*)dt);
```

### Hiperparametre Optimizasyonu

```c
#include "model_selection.h"

ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){5, 10, 15, 20}, 4);
param_grid_add_int(&grid, "min_samples_split", (int[]){2, 5, 10}, 3);

DecisionTreeClassifier *dt = decision_tree_classifier_create();
GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);

gs->base.fit((Estimator*)gs, X, y);
printf("Optimal Skor: %.4f\n", gs->best_score_);

grid_search_cv_free(gs);
param_grid_free(&grid);
```

### Boyut İndirgeme

```c
#include "decomposition.h"

PCA *pca = pca_create(2);
pca->base.fit((Estimator*)pca, X, NULL);

Matrix *X_projected = pca->base.transform((Estimator*)pca, X);

const double *explained_variance = pca_explained_variance_ratio(pca);
printf("Kümülatif Açıklanan Varyans: %.2f%%\n",
       (explained_variance[0] + explained_variance[1]) * 100);

pca->base.free((Estimator*)pca);
```

### Özellik Seçimi

```c
#include "feature_selection.h"

// İstatistiksel özellik sıralaması
SelectKBest *selector = select_k_best_create(SCORE_F_REGRESSION, 10);
selector->base.fit((Estimator*)selector, X, y);
Matrix *X_selected = selector->base.transform((Estimator*)selector, X);

// Varyans tabanlı filtreleme
VarianceThreshold *vt = variance_threshold_create(0.01);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);

selector->base.free((Estimator*)selector);
vt->base.free((Estimator*)vt);
```

## Proje Mimarisi

```
tinycml/
├── include/                # Genel başlık dosyaları
│   ├── matrix.h            # Matris ve vektör işlemleri
│   ├── estimator.h         # Temel tahmin edici arayüzü
│   ├── pipeline.h          # Pipeline altyapısı
│   ├── validation.h        # Çapraz doğrulama araçları
│   ├── model_selection.h   # GridSearchCV uygulaması
│   ├── linear_regression.h
│   ├── logistic_regression.h
│   ├── knn.h
│   ├── kmeans.h
│   ├── naive_bayes.h
│   ├── decision_tree.h
│   ├── ensemble.h          # Rastgele Orman
│   ├── neural_network.h
│   ├── decomposition.h     # PCA
│   ├── feature_selection.h
│   ├── preprocessing.h
│   └── metrics.h
├── src/                    # Uygulama dosyaları
├── examples/               # Çalıştırılabilir gösterimler
├── tests/                  # Birim test paketi
├── data/                   # Örnek veri setleri
└── docs/                   # API dokümantasyonu
```

## Lisans

Bu yazılım MIT Lisansı altında dağıtılmaktadır.
