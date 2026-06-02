# TinyCML WebAssembly Build

Run tinycml models directly in the browser with near-native performance.

## Prerequisites

Install [Emscripten](https://emscripten.org/):

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

## Build

```bash
cd wasm
make
```

This produces:
- `tinycml.js` — JavaScript glue code
- `tinycml.wasm` — WebAssembly binary

## Demo

Serve the demo with any static file server:

```bash
# Python
python3 -m http.server 8080

# Node.js
npx serve .

# Then open http://localhost:8080/demo.html
```

The demo provides:
- 5 dataset types (moons, circles, XOR, linear, spiral)
- 4 models (Logistic Regression, KNN, Gaussian NB, Decision Tree)
- Interactive decision boundary visualization
- Training time and accuracy display

## API Reference

When compiled with `-s MODULARIZE=1`, tinycml exposes:

```javascript
const TinyCML = require('./tinycml.js');
const cml = await TinyCML();

// Allocate a 3x2 matrix
const mat = cml._matrix_alloc(3, 2);

// Set values
cml._matrix_set(mat, 0, 0, 1.5);
cml._matrix_set(mat, 1, 0, 2.3);

// Free
cml._matrix_free(mat);
```

### Exported Functions

| Function | Description |
|----------|-------------|
| `_matrix_alloc(rows, cols)` | Create matrix |
| `_matrix_free(m)` | Free matrix |
| `_matrix_set(m, i, j, val)` | Set element |
| `_matrix_get(m, i, j)` | Get element |
| `_logistic_regression_create(lr, max_iter)` | Create logreg |
| `_logistic_regression_fit(m, X, y)` | Train logreg |
| `_logistic_regression_predict(m, X)` | Predict |
| `_knn_fit(X, y, k)` | Train KNN |
| `_knn_predict(X, model, k)` | Predict KNN |
| `_gaussian_nb_create()` | Create Naive Bayes |
| `_gaussian_nb_fit(m, X, y)` | Train NB |
| `_gaussian_nb_predict(m, X)` | Predict NB |

## Build Options

```bash
# Debug build with assertions
make CFLAGS="-O0 -g"

# Release build with optimizations
make CFLAGS="-O3 -flto"

# Single-header build
make USE_SINGLE_HEADER=1
```

## Performance Tips

- Use `-O3 -flto` for production builds
- Set `-s ALLOW_MEMORY_GROWTH=0` with fixed heap if possible
- Use `-s INITIAL_MEMORY=16777216` for 16MB initial heap
- Avoid frequent malloc/free — pre-allocate buffers
