#pragma once
#include <model.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <LearningRateScheduler.hpp>
#include <dataset.hpp>
#include <optimizer.hpp>

// Linear Regression Model
// This class implements a simple linear regression model using Eigen for matrix operations.
// It includes methods for fitting the model to training data, making predictions, and updating parameters.

class LinearRegression : public Model
{
private:
    Eigen::VectorXd weights_; // [n_features]
    double bias_ = 0.0;
    double learning_rate_;
    int epochs_;
    int batch_size_;

public:
    LinearRegression(double lr = 0.001, int epochs = 1000, int batch_size = 32)
        : learning_rate_(lr), epochs_(epochs), batch_size_(batch_size) {
        if (lr <= 0.0) throw std::invalid_argument("Learning rate must be positive.");
        if (epochs <= 0) throw std::invalid_argument("Number of epochs must be positive.");
        if (batch_size <= 0) throw std::invalid_argument("Batch size must be positive.");
        weights_ = Eigen::VectorXd::Zero(0);
    }

    void fit(const Dataset &train) override {
        weights_ = Eigen::VectorXd::Zero(train.getNumFeatures());
        bias_ = 0.0;
        
        // Create optimizer and loss function
        GradientDescent optimizer(learning_rate_);
        MeanSquaredError loss;
        // Use exponential decay learning rate scheduler with faster decay
        ExponentialDecayLearningRateScheduler scheduler(learning_rate_, 0.01);
        
        // Train using SGD
        optimizer.optimize(*this, train, loss, scheduler, epochs_, batch_size_);
        std::cout << "Model trained successfully using SGD." << std::endl;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const override {
        if (weights_.size() == 0) throw std::runtime_error("Model has not been trained yet. Call fit() before predict().");
        return X * weights_ + Eigen::VectorXd::Constant(X.rows(), bias_);
    }

    void update_parameters(Eigen::VectorXd gradients, double rate) override {
        if (weights_.size() == 0) throw std::runtime_error("Model has not been trained yet. Call fit() before update_parameters().");
        
        // Split gradients into weights and bias updates
        // The last element of gradients is for bias
        Eigen::VectorXd weight_gradients = gradients.head(weights_.size());
        double bias_gradient = gradients.tail(1)(0);
        
        // Update parameters with bounds to prevent explosion
        weights_ -= rate * weight_gradients;
        bias_ -= rate * bias_gradient;
        
        // Clip weights to prevent explosion
        double max_weight = 10.0;
        weights_ = weights_.cwiseMax(-max_weight).cwiseMin(max_weight);
        
        // Clip bias to prevent explosion
        double max_bias = 10.0;
        bias_ = std::max(-max_bias, std::min(max_bias, bias_));
    }

    std::string name() const override { return "Linear Regression"; }
    std::string description() const override { return "A simple linear regression model."; }
    std::string formula() const override { return "y = Xw + b"; }
    std::string gradient_formula() const override { return "∇L = -2/n * X^T(y - Xw)"; }
    Eigen::VectorXd get_weights() const { return weights_; }
    double get_bias() const { return bias_; }
    ~LinearRegression() override = default;
};
