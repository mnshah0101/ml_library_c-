#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include "csv_loader.hpp"


//load csv into a csv loader object
//than add to data class


class Dataset {
    private:
        Eigen::MatrixXd X_;
        Eigen::VectorXd y_;

    public: 

    Dataset(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) : X_(X), y_(y) {}

    Eigen::MatrixXd getX() const { return X_; }
    Eigen::VectorXd getY() const { return y_; }
    int getNumRows() const { return X_.rows(); }
    int getNumFeatures() const { return X_.cols(); }
    Dataset shuffle(unsigned int seed) const {
        std::cout << "Shuffling with seed: " << seed << std::endl;
        std::vector<int> indices(X_.rows());
        for (int i = 0; i < X_.rows(); ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
        
        // Print first few indices to verify shuffling
        std::cout << "First 5 indices after shuffle: ";
        for (int i = 0; i < std::min(5, (int)indices.size()); ++i) {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;
        
        // Create new matrices for shuffled data
        Eigen::MatrixXd X_shuffled(X_.rows(), X_.cols());
        Eigen::VectorXd y_shuffled(X_.rows());
        
        // Copy data in shuffled order
        for (int i = 0; i < X_.rows(); ++i) {
            X_shuffled.row(i) = X_.row(indices[i]);
            y_shuffled(i) = y_(indices[i]);
        }
        
        return Dataset(X_shuffled, y_shuffled);
    }

    std::pair<Dataset, Dataset> trainTestSplit(double test_size = 0.2, uint seed = std::random_device{}()) {
        std::cout << "Performing train-test split with seed: " << seed << std::endl;
        // First shuffle the data
        Dataset shuffled = shuffle(seed);
        
        // Then split the dataset into training and testing sets
        int num_rows = shuffled.getNumRows();
        int num_train = static_cast<int>(num_rows * (1 - test_size));
        Dataset train_set(shuffled.getX().topRows(num_train), shuffled.getY().head(num_train));
        Dataset test_set(shuffled.getX().bottomRows(num_rows - num_train), shuffled.getY().tail(num_rows - num_train));
        
        std::cout << "Split into " << num_train << " training samples and " 
                  << (num_rows - num_train) << " test samples" << std::endl;
        return std::make_pair(train_set, test_set);
    }

    void saveToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        // Write header
        for (int i = 0; i < X_.cols(); i++) {
            file << "X" << i;
            if (i < X_.cols() - 1) {
                file << ",";
            }
        }
        file << ",y\n";
        // Write data
        for (int i = 0; i < X_.rows(); i++) {
            for (int j = 0; j < X_.cols(); j++) {
                file << X_(i, j);
                if (j < X_.cols() - 1) {
                    file << ",";
                }
            }
            file << "," << y_(i) << "\n";
        }
    }

    void print() const {
        std::cout << "X:\n" << X_ << "\n";
        std::cout << "y:\n" << y_.transpose() << "\n";
    }
};

// Add toDataset as a free function
inline Dataset toDataset(const CSVLoader &loader,
                         const std::vector<std::string> &feature_columns,
                         const std::string &target_column)
{
    const auto &m_data = loader.getData();
    if (m_data.empty())
    {
        throw std::runtime_error("No data loaded. Call load() first.");
    }

    // Get indices for features and target
    std::vector<int> feature_indices;
    for (const auto &col : feature_columns)
    {
        feature_indices.push_back(loader.getColumnIndex(col));
    }
    int target_index = loader.getColumnIndex(target_column);

    // First pass: count valid rows
    int valid_rows = 0;
    for (int i = 1; i < m_data.size(); ++i)
    { // Start from 1 to skip header
        bool is_valid = true;
        try
        {
            // Check if target value is valid
            std::stod(m_data[i][target_index]);
            // Check if all feature values are valid
            for (int idx : feature_indices)
            {
                std::stod(m_data[i][idx]);
            }
            valid_rows++;
        }
        catch (const std::exception &)
        {
            // Skip this row if any value is invalid
            continue;
        }
    }

    // Create matrices with correct size
    Eigen::MatrixXd X(valid_rows, feature_indices.size());
    Eigen::VectorXd y(valid_rows);

    // Second pass: fill matrices with valid data
    int current_row = 0;
    for (int i = 1; i < m_data.size(); ++i) {  // Start from 1 to skip header
        try {
            // Check target value
            if (current_row >= valid_rows) {
                continue;
            }
            double target_val = std::stod(m_data[i][target_index]);
            // Check and store feature values
            for (int j = 0; j < feature_indices.size(); ++j) {
                X(current_row, j) = std::stod(m_data[i][feature_indices[j]]);
            }
            y(current_row) = target_val;
            current_row++;
        } catch (const std::exception&) {
            // Skip this row if any value is invalid
            continue;
        }
    }


    return Dataset(X, y);
}

// Keep the old version for backward compatibility
inline Dataset toDataset(const CSVLoader& loader, int target_column = -1) {
    const auto& m_data = loader.getData();
    if (m_data.empty()) {
        throw std::runtime_error("No data loaded. Call load() first.");
    }

    // If target_column is -1, use the last column as target
    if (target_column == -1) {
        target_column = m_data[0].size() - 1;
    }

    // First pass: count valid rows
    int valid_rows = 0;
    for (int i = 1; i < m_data.size(); ++i) {  // Start from 1 to skip header
        bool is_valid = true;
        try {
            // Check if target value is valid
            std::stod(m_data[i][target_column]);
            // Check if all feature values are valid
            for (int j = 0; j < m_data[i].size(); ++j) {
                if (j != target_column) {
                    std::stod(m_data[i][j]);
                }
            }
            valid_rows++;
        } catch (const std::exception&) {
            // Skip this row if any value is invalid
            std::cout << "Skipping row " << i << " due to invalid values." << std::endl;
            continue;
        }
    }

    std::cout << "Total valid rows found: " << valid_rows << std::endl;
    std::cout << "Total data rows: " << m_data.size() - 1 << std::endl;  // -1 for header

    // Create matrices with correct size
    int num_features = m_data[0].size() - 1;  // Excluding target column
    std::cout << "Number of features: " << num_features << std::endl;
    Eigen::MatrixXd X(valid_rows, num_features);
    Eigen::VectorXd y(valid_rows);

    // Second pass: fill matrices with valid data
    int current_row = 0;
    for (int i = 1; i < m_data.size(); ++i) {  // Start from 1 to skip header
        try {
            int feature_idx = 0;
            for (int j = 0; j < m_data[i].size(); ++j) {
                if (j == target_column) {
                    y(current_row) = std::stod(m_data[i][j]);
                } else if (feature_idx < num_features) {  // Add bounds check
                    X(current_row, feature_idx) = std::stod(m_data[i][j]);
                    feature_idx++;
                }
            }
            current_row++;
            if (current_row % 100 == 0) {  // Log progress every 100 rows
                std::cout << "Processed " << current_row << " rows" << std::endl;
            }
        } catch (const std::exception&) {
            // Skip this row if any value is invalid
            std::cout << "Error processing row " << i << std::endl;
            continue;
        }
    }

    std::cout << "Final current_row: " << current_row << std::endl;
    std::cout << "Matrix X dimensions: " << X.rows() << "x" << X.cols() << std::endl;
    std::cout << "Vector y dimensions: " << y.size() << std::endl;

    // Feature scaling (standardization)
    for (int j = 0; j < X.cols(); ++j) {
        double mean = X.col(j).mean();
        double std = std::sqrt((X.col(j).array() - mean).square().mean());
        if (std > 1e-10) {  // Avoid division by zero
            X.col(j) = (X.col(j).array() - mean) / std;
        }
    }

    // Target scaling
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().mean());
    if (y_std > 1e-10) {
        y = (y.array() - y_mean) / y_std;
    }

    return Dataset(X, y);
}
