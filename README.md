# C++ Machine Learning Library

A modern C++ machine learning library that provides implementations of various machine learning algorithms using Eigen for efficient matrix operations.

## Features

### Data Handling
- **CSVLoader**: Load and parse CSV files with support for:
  - Custom delimiters
  - Column name mapping
  - Data validation
  - Whitespace trimming
  - Quote handling

- **Dataset**: Manage and manipulate datasets with features:
  - Train-test splitting
  - Data shuffling
  - Feature scaling (standardization)
  - CSV export
  - Dimension information

### Machine Learning Models

#### Supervised Learning
1. **Linear Regression**
   - Gradient descent optimization
   - Batch processing
   - Learning rate scheduling
   - Mean squared error loss

2. **Logistic Regression**
   - Binary classification
   - Sigmoid activation
   - Stochastic gradient descent
   - Batch processing support

3. **K-Nearest Neighbors (KNN)**
   - Configurable number of neighbors
   - Euclidean distance metric
   - Efficient prediction for single instances
   - Support for both regression and classification

4. **Decision Tree**
   - Configurable maximum depth
   - Gini impurity and entropy metrics
   - Information gain splitting
   - Support for both regression and classification

#### Unsupervised Learning
1. **K-Means Clustering**
   - Configurable number of clusters
   - Maximum iterations limit
   - Random centroid initialization
   - Convergence detection

2. **Principal Component Analysis (PCA)**
   - Dimensionality reduction
   - Explained variance calculation
   - Data transformation and inverse transformation
   - Component analysis

### Optimization
- **Gradient Descent Optimizer**
  - Configurable learning rate
  - Batch shuffling
  - Learning rate scheduling
  - Support for custom loss functions

### Learning Rate Scheduling
- **Exponential Decay Scheduler**
  - Configurable initial learning rate
  - Minimum learning rate threshold
  - Automatic decay rate adjustment

## Dependencies
- Eigen (for matrix operations)
- C++17 or later
- Standard C++ libraries
- CMake 3.10 or later

## Installation

### Building from Source

1. Install Eigen:
```bash
# On macOS
brew install eigen

# On Ubuntu/Debian
sudo apt-get install libeigen3-dev
```

2. Build and install the library:
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

This will install:
- The static library (`libml_library.a`)
- Header files in `/usr/local/include/ml_library/`
- CMake configuration files in `/usr/local/lib/cmake/ml_library/`

### Using the Library in Your Project

1. Add to your project's CMakeLists.txt:
```cmake
find_package(ml_library REQUIRED)
target_link_libraries(your_target PRIVATE ml_library::ml_library)
```

2. Include the headers in your code:
```cpp
#include <ml_library/KMeans.hpp>
#include <ml_library/PCA.hpp>
// ... other headers as needed
```

## Usage Example

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <ml_library/csv_loader.hpp>
#include <ml_library/dataset.hpp>
#include <ml_library/LinearRegression.hpp>

int main() {
    // Load data
    CSVLoader loader("data/example.csv");
    loader.load();

    // Select features and target
    std::vector<std::string> feature_columns = {"feature1", "feature2"};
    std::string target_column = "target";

    // Convert to dataset
    Dataset dataset = toDataset(loader, feature_columns, target_column);
    
    // Split into training and testing sets
    auto [train_set, test_set] = dataset.trainTestSplit(0.8);
    
    // Create and train model
    LinearRegression model(0.01, 1000, 32);
    model.fit(train_set);
    
    // Make predictions
    Eigen::VectorXd predictions = model.predict(test_set.getX());
    
    return 0;
}
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
