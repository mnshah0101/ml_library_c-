#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iomanip>

#include "../include/KMeans.hpp"
#include "../include/PCA.hpp"
#include "../include/csv_loader.hpp"
#include "../include/dataset.hpp"

void printMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    std::cout << "\n" << name << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for(int i = 0; i < matrix.rows(); i++) {
        for(int j = 0; j < matrix.cols(); j++) {
            std::cout << std::setw(10) << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(const Eigen::VectorXd& vector, const std::string& name) {
    std::cout << "\n" << name << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for(int i = 0; i < vector.size(); i++) {
        std::cout << std::setw(10) << vector(i) << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        // Load data
        CSVLoader loader("data/titanic_clean.csv");
        loader.load();

        // Print available columns
        std::cout << "Available columns:" << std::endl;
        for (const auto& col : loader.getColumnNames()) {
            std::cout << "- " << col << std::endl;
        }

        // Select features and target
        std::vector<std::string> feature_columns = {
           "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
        };
        std::string target_column = "Survived";

        // Convert to dataset
        Dataset dataset = toDataset(loader, feature_columns, target_column);
        
        // Print dataset dimensions
        std::cout << "\nDataset dimensions:" << std::endl;
        std::cout << "X shape: " << dataset.getX().rows() << "x" << dataset.getX().cols() << std::endl;
        std::cout << "y shape: " << dataset.getY().size() << std::endl;

        // Split into training and testing sets
        auto [train_set, test_set] = dataset.trainTestSplit(0.8);
        
        // Print train/test set dimensions
        std::cout << "\nTrain set dimensions:" << std::endl;
        std::cout << "X shape: " << train_set.getX().rows() << "x" << train_set.getX().cols() << std::endl;
        std::cout << "y shape: " << train_set.getY().size() << std::endl;
        
        std::cout << "\nTest set dimensions:" << std::endl;
        std::cout << "X shape: " << test_set.getX().rows() << "x" << test_set.getX().cols() << std::endl;
        std::cout << "y shape: " << test_set.getY().size() << std::endl;

        // PCA Test
        std::cout << "\n=== PCA Test ===" << std::endl;
        
        // Create and fit PCA model
        PCA pca(2);  // Reduce to 2 dimensions
        pca.fit(train_set);
        
        // Transform the data
        Eigen::MatrixXd transformed_train = pca.transform(train_set.getX());
        Eigen::MatrixXd transformed_test = pca.transform(test_set.getX());
        
        // Print results
        std::cout << "\nOriginal feature count: " << train_set.getX().cols() << std::endl;
        std::cout << "Reduced feature count: " << transformed_train.cols() << std::endl;
        
        // Print explained variance ratio
        printVector(pca.get_explained_variance_ratio(), "Explained Variance Ratio");
        
        // Print first few transformed samples
        printMatrix(transformed_train.block(0, 0, 5, 2), "First 5 Transformed Training Samples");
        
        // Test inverse transform
        Eigen::MatrixXd reconstructed = pca.inverse_transform(transformed_train);
        printMatrix(reconstructed.block(0, 0, 5, reconstructed.cols()), "First 5 Reconstructed Samples");
        
        // Calculate reconstruction error
        double reconstruction_error = (train_set.getX() - reconstructed).norm() / train_set.getX().norm();
        std::cout << "\nReconstruction Error: " << reconstruction_error << std::endl;

        // KMeans Test
        std::cout << "\n=== KMeans Test ===" << std::endl;
        
        // Create and train KMeans model
        KMeans model(3, 100);
        model.fit(transformed_train);  // Using PCA-transformed data

        // Make predictions
        Eigen::VectorXi predictions = model.predict(transformed_test);

        // Print predictions
        std::cout << "\nPredictions:" << std::endl;
        for (int i = 0; i < predictions.size(); i++) {
            std::cout << "Sample " << i << ": " << predictions(i) << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 
