#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "../include/LogisticRegression.hpp"
#include "../include/csv_loader.hpp"
#include "../include/dataset.hpp"

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

        // Create and train model
        LogisticRegression model;
        model.fit(train_set);

        // Make predictions
        Eigen::VectorXd predictions = model.predict(test_set.getX());
        
        // Print prediction dimensions
        std::cout << "\nPrediction dimensions:" << std::endl;
        std::cout << "Predictions shape: " << predictions.size() << std::endl;
        std::cout << "Test set y shape: " << test_set.getY().size() << std::endl;

        // Print predictions

        // Calculate and print metrics
        double mse = (predictions - test_set.getY()).array().square().mean();
        double rmse = std::sqrt(mse);
        double mae = (predictions - test_set.getY()).array().abs().mean();

        std::cout << "\nModel Performance:" << std::endl;
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "RMSE: " << rmse << std::endl;
        std::cout << "MAE: " << mae << std::endl;

        // Print feature importance
        std::cout << "\nFeature Importance:" << std::endl;
        for (size_t i = 0; i < feature_columns.size(); ++i) {
            std::cout << feature_columns[i] << ": " << model.get_weights()(i) << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 
