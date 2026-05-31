# tinycml Development Roadmap

## Phase 1: Cleanup & Bug Fixes ✅
- [x] `strdup` → `cml_strdup` (ISO C11 compliance)
- [x] DecisionTree: bubble sort → qsort (O(N³)→O(N²))
- [x] GridSearchCV generic `apply_params()` (DT + NN)
- [x] NN global RNG reseed removed (struct-local seed)
- [x] NN backward NULL checks (`d_w`, `delta_back`)
- [x] `fread` return checks in `linear_regression.c`
- [x] README ⏳ markers updated

## Phase 2: Test Infrastructure ✅
- [x] 13 test suites / 66 tests total
- [x] Matrix (9), LinearRegression (3), LogisticRegression (4), KNN (5)
- [x] KMeans (5), Metrics (12), DecisionTree (5), NaiveBayes (4)
- [x] NeuralNetwork (4), PCA (4), Preprocessing (5), SVM (3), OneHotEncoder (3)
- [x] NULL input safety tests across all modules

## Phase 3: Performance Optimizations ✅
- [x] Matrix matmul: cache-friendly B-transpose + i-k-j loop (2-5× speedup)
- [x] xoshiro256\*\* PRNG replacing `srand`/`rand`
- [x] DecisionTree split buffer pre-allocation (no malloc in inner loop)

## Phase 4: Missing Algorithms ✅
- [x] Gaussian Naive Bayes (Estimator interface, log-sum-exp proba)
- [x] Linear SVM (sub-gradient descent, hinge loss, {0,1}→{-1,+1} conversion)
- [x] OneHotEncoder (fit unique categories + transform to binary)

## Phase 5: CI/CD & Distribution ✅
- [x] Makefile: static (`libtinycml.a`) + shared (`libtinycml.so`) libraries
- [x] `make install` / `make uninstall` targets (`PREFIX=/usr/local`)
- [x] pkg-config template (`tinycml.pc.in`)
- [x] GitHub Actions CI: gcc/clang matrix + ASAN/UBSan + Doxygen
- [x] Doxyfile for API documentation
- [x] README updated (metrics, ⏳ removed)

## Phase 6: Bug Fixes & Estimator Integration ✅
- [x] Multi-class metrics (confusion_matrix_multi, precision/recall/f1 macro & weighted)
- [x] Estimator integration for LogisticRegression
- [x] Expanded test coverage (KNN/SVM predict_proba)

## Phase 7: Advanced Algorithms ✅
- [x] OpenMP parallel matmul (guarded, zero-cost when disabled)
- [x] Softmax Regression (multinomial logistic, cross-entropy)
- [x] Multinomial Naive Bayes (Laplace smoothing, count-based)
- [x] RBF SVM (kernel trick, Platt scaling for probability)
- [x] DBSCAN (density-based clustering, noise detection)
- [x] Ridge Regression (L2-regularized, closed-form)
- [x] Lasso Regression (L1-regularized, coordinate descent)

## Phase 8: Production Quality ✅
- [x] Unified error system (`cml_error.h`, thread-local error state)
- [x] Model serialization (binary save/load for LR/SVM/NB/LogReg)
- [x] Robust CSV parser (quoted fields, missing values, auto-detect delimiter)

## Phase 9: Clustering Metrics & Build Improvements ✅
- [x] Silhouette Score (Euclidean distance, per-sample a(i)/b(i))
- [x] Shared library soname versioning (libtinycml.so.0.1.0 → .so.0 → .so)
- [x] Test warning fixes (sign-compare in pipeline tests)

## Completed: All Phases Done 🎉

- **27 source files** (src/*.c)
- **27 headers** (include/*.h)
- **30 test suites / 127 tests**
- **11,600+ LOC**
- **0 external dependencies**
- **ISO C11 compliant**

### Optional Future Enhancements
- [ ] Optional BLAS backend for large matrices
- [ ] CMake alternative build system (basic CMakeLists.txt exists)
- [ ] Python bindings via ctypes/cffi
- [ ] ARM NEON / SIMD intrinsics for matrix operations
