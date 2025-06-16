#pragma once
#include "model.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include "LearningRateScheduler.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

// Linear Regression Model
// This class implements a simple linear regression model using Eigen for matrix operations.
// It includes methods for fitting the model to training data, making predictions, and updating parameters.

class KNearestNeighbors : public Model
{
    private:
        Dataset _data;
        int _k = 3; // Number of neighbors to consider, default is 3

        double EuclideanDistance(const Eigen::RowVectorXd &a, const Eigen::RowVectorXd &b) const
        {
            return (a - b).norm(); // Calculate the Euclidean distance between two vectors
        }

        
    public:

        KNearestNeighbors(int k = 3) : _k(k), _data(Eigen::MatrixXd(), Eigen::VectorXd())
        {
            if (k <= 0)
            {
                throw std::invalid_argument("Number of neighbors k must be a positive integer.");
            }
        }

        void fit(const Dataset &train) override
        {
            _data = Dataset(train.getX(), train.getY());
        }

        // Fit the model to the training data
        void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) 
        {
            // Store the training data
            _data = Dataset(X, y);
        }

        // Predict the target values for the given input features
        Eigen::VectorXd predict(const Eigen::MatrixXd &X) const override
        {
            if (X.rows() == 0 || X.cols() != _data.getNumFeatures())
            {
                throw std::invalid_argument("Input matrix dimensions do not match training data.");
            }
            Eigen::VectorXd predictions(X.rows());
            for (int i = 0; i < X.rows(); ++i)
            {
                // For each input row, find the k nearest neighbors and predict the target value
                predictions(i) = predictSingle(X.row(i));
            }
            return predictions;
        }

        // Predict a single instance based on the nearest neighbors
        double predictSingle(const Eigen::RowVectorXd &x) const
        {
            if (_data.getNumRows() == 0) 
            {
                throw std::runtime_error("Model has not been trained with any data.");
            }

            // Create a vector to store distances and corresponding indices
            std::vector<std::pair<double, int>> distances;

            // Calculate distances from the input vector to all training data points
            for (int i = 0; i < _data.getNumRows(); ++i)
            {
                double dist = EuclideanDistance(x, _data.getX().row(i));
                distances.emplace_back(dist, i);
            }

            // Sort distances to find the k nearest neighbors
            std::sort(distances.begin(), distances.end());

            // Calculate the mean of the target values of the k nearest neighbors
            double sum = 0.0;
            for (int i = 0; i < _k && i < distances.size(); ++i)
            {
                sum += _data.getY()(distances[i].second);
            }
            return sum / std::min(_k, static_cast<int>(distances.size())); // Handle case where there are fewer than k neighbors

        }

        // Update model parameters (not applicable for KNN, but required by the Model interface)
        void update_parameters(Eigen::VectorXd gradients, double rate) override
        {
            throw std::logic_error("KNearestNeighbors does not support parameter updates.");
        }

        // Return the name of the model
        std::string name() const override
        {
            return "KNearestNeighbors";
        }
        
        // Return a description of the model
        std::string description() const override
        {
            return "K-Nearest Neighbors regression model.";
        }

        // Return the formula used by the model (not applicable for KNN)
        std::string formula() const override
        {
            return "y = mean(y_neighbors) for k nearest neighbors";
        }
        
        // Return the gradient formula (not applicable for KNN)
        std::string gradient_formula() const override
        {
            return "Not applicable for KNN";
        }   

        // Destructor
        ~KNearestNeighbors() override = default;

};

