# Roadmap / Yol Haritası

[English](#english) | [Türkçe](#türkçe)

---

# English

## tinycml Development Status

### Completed Features ✅

#### Neural Networks
- [x] Feedforward neural network with configurable layers
- [x] Activation functions (ReLU, sigmoid, tanh, softmax)
- [x] Backpropagation algorithm
- [x] Mini-batch gradient descent

#### Decision Trees & Ensembles
- [x] Decision tree algorithm with Gini/Entropy criteria
- [x] Tree depth and sample constraints
- [x] Random Forest ensemble with bootstrap
- [x] Out-of-Bag (OOB) score calculation

#### Additional Algorithms
- [x] Naive Bayes classifier (Gaussian)
- [x] Support Vector Machine (linear kernel)
- [x] Principal Component Analysis (PCA) with whitening
- [x] Regularization (L2) for linear models

#### Features & Infrastructure
- [x] Unified Estimator API (fit/predict/score)
- [x] Pipeline system for chaining transformers
- [x] Model serialization (save/load to binary file)
- [x] Cross-validation utilities (k-fold, stratified)
- [x] GridSearchCV for hyperparameter tuning
- [x] Learning curves (training history to CSV)
- [x] Verbose output and training callbacks
- [x] Feature selection (SelectKBest, VarianceThreshold)
- [x] Scoring functions (f_classif, f_regression, chi2, mutual_info)

#### Preprocessing
- [x] StandardScaler, MinMaxScaler
- [x] One-hot encoding for categorical variables
- [x] Polynomial feature expansion

### Future Enhancements 🚀

#### Performance Optimizations
- [ ] SIMD optimizations for matrix operations (SSE/AVX)
- [ ] Parallel processing with OpenMP
- [ ] Memory pool for matrix allocations
- [ ] Cache-friendly matrix multiplication (blocked/tiled)

#### Additional Features
- [ ] Dropout regularization for neural networks
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Missing value imputation
- [ ] Outlier detection and handling
- [ ] Multinomial Naive Bayes
- [ ] Gradient Boosting

#### Documentation
- [ ] Algorithm theory documentation with math
- [ ] Performance benchmarks
- [ ] Comparison with scikit-learn results
- [ ] Video tutorials

## Contributing

Contributions are welcome! Here's how you can help:

1. **Bug fixes**: Submit issues and pull requests
2. **New algorithms**: Implement algorithms from the roadmap
3. **Documentation**: Improve explanations and examples
4. **Testing**: Add more unit tests and edge cases
5. **Performance**: Profile and optimize critical paths

---

# Türkçe

## tinycml Geliştirme Durumu

### Tamamlanan Özellikler ✅

#### Sinir Ağları
- [x] Yapılandırılabilir katmanlara sahip ileri beslemeli sinir ağı
- [x] Aktivasyon fonksiyonları (ReLU, sigmoid, tanh, softmax)
- [x] Geri yayılım algoritması
- [x] Mini-batch gradient descent

#### Karar Ağaçları ve Topluluklar
- [x] Gini/Entropi kriterleriyle karar ağacı algoritması
- [x] Ağaç derinliği ve örnek kısıtlamaları
- [x] Bootstrap ile Random Forest topluluk yöntemi
- [x] Out-of-Bag (OOB) skor hesaplaması

#### Ek Algoritmalar
- [x] Naive Bayes sınıflandırıcı (Gaussian)
- [x] Destek Vektör Makinesi (lineer çekirdek)
- [x] Beyazlatma ile Temel Bileşen Analizi (PCA)
- [x] Lineer modeller için düzenlileştirme (L2)

#### Özellikler ve Altyapı
- [x] Birleşik Estimator API'si (fit/predict/score)
- [x] Dönüştürücüleri zincirleme için Pipeline sistemi
- [x] Model serileştirme (ikili dosyaya kaydet/yükle)
- [x] Çapraz doğrulama araçları (k-katlı, katmanlı)
- [x] Hiperparametre ayarı için GridSearchCV
- [x] Öğrenme eğrileri (eğitim geçmişini CSV'ye)
- [x] Ayrıntılı çıktı ve eğitim callback'leri
- [x] Özellik seçimi (SelectKBest, VarianceThreshold)
- [x] Puanlama fonksiyonları (f_classif, f_regression, chi2, mutual_info)

#### Ön İşleme
- [x] StandardScaler, MinMaxScaler
- [x] Kategorik değişkenler için one-hot kodlama
- [x] Polinom özellik genişletme

### Gelecek Geliştirmeler 🚀

#### Performans Optimizasyonları
- [ ] Matris işlemleri için SIMD optimizasyonları (SSE/AVX)
- [ ] OpenMP ile paralel işleme
- [ ] Matris ayırmaları için bellek havuzu
- [ ] Önbellek dostu matris çarpımı (bloklu/döşemeli)

#### Ek Özellikler
- [ ] Sinir ağları için Dropout düzenlileştirme
- [ ] Lineer Diskriminant Analizi (LDA)
- [ ] Eksik değer doldurma
- [ ] Aykırı değer tespiti ve işleme
- [ ] Multinomial Naive Bayes
- [ ] Gradient Boosting

#### Dokümantasyon
- [ ] Matematik ile algoritma teorisi dokümantasyonu
- [ ] Performans karşılaştırmaları
- [ ] scikit-learn sonuçlarıyla karşılaştırma
- [ ] Video eğitimleri

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! İşte nasıl yardımcı olabileceğiniz:

1. **Hata düzeltmeleri**: Issue ve pull request gönderin
2. **Yeni algoritmalar**: Yol haritasındaki algoritmaları uygulayın
3. **Dokümantasyon**: Açıklamaları ve örnekleri iyileştirin
4. **Test**: Daha fazla birim testi ve uç durum ekleyin
5. **Performans**: Kritik yolları profil edin ve optimize edin
