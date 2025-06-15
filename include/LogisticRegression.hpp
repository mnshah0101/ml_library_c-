#pragma once
#include <model.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <LearningRateScheduler.hpp>
#include <dataset.hpp>
#include <optimizer.hpp>

class LogisticRegression : public Model
{
private:
    Eigen::VectorXd weights_;
    Eigen::VectorXd bias_;
    double lr_{};
    int epochs_{};
    int batch_size_{};


    // Sigmoid function
    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    // Sigmoid function for vector
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd &z) {
        return (1.0 + (-z.array()).exp()).inverse();
    }

public:
    LogisticRegression(double lr = 0.01, int epochs = 1000, int batch_size = 32): lr_{lr}, epochs_{epochs}, batch_size_{batch_size}{
        if (lr <= 0.0)
            throw std::invalid_argument("Learning rate must be positive.");
        if (epochs <= 0)
            throw std::invalid_argument("Number of epochs must be positive.");
        if (batch_size <= 0)
            throw std::invalid_argument("Batch size must be positive.");
        weights_ = Eigen::VectorXd::Zero(0);
    }



    void fit(const Dataset &train) override{
        weights_ = Eigen::VectorXd::Zero(train.getNumFeatures());
        bias_ = Eigen::VectorXd::Zero(1);
        Eigen::MatrixXd X = train.getX();
        Eigen::VectorXd y = train.getY();
        if (X.rows() == 0 || X.cols() == 0)
            throw std::runtime_error("Training data is empty.");
        if (y.size() != X.rows())
            throw std::runtime_error("Mismatch between number of samples in X and y.");
        for (int epoch = 0; epoch < epochs_; ++epoch) {
            for (int i = 0; i < X.rows(); i += batch_size_) {
                auto z = weights_.dot(X.row(i).transpose()) + bias_(0);
                auto prediction = sigmoid(z);
                auto error = prediction - y(i);
                
                // Update weights and bias
                weights_ -= lr_ * error * X.row(i).transpose();
                bias_ -= lr_ * Eigen::VectorXd::Constant(1, error);
            }
        }
        std::cout << "Model trained successfully." << std::endl;
    }

    void update_parameters(Eigen::VectorXd gradients, double rate) override {
        if (weights_.size() == 0)
            throw std::runtime_error("Model has not been trained yet. Call fit() before update_parameters().");
        
        // Split gradients into weights and bias updates
        Eigen::VectorXd weight_gradients = gradients.head(weights_.size());
        double bias_gradient = gradients.tail(1)(0);
        
        // Update parameters with bounds to prevent explosion
        weights_ -= rate * weight_gradients;
        bias_(0) -= rate * bias_gradient;   


    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const override // returns P(class=1)
    {
        if (weights_.size() == 0)
            throw std::runtime_error("Model has not been trained yet. Call fit() before predict().");
        Eigen::VectorXd z = X * weights_ + bias_.replicate(X.rows(), 1);
        return sigmoid(z);
    }

    std::string name() const override {
        return "Logistic Regression";
    }

    std::string description() const override {
        return "Logistic Regression model for binary classification.";
    }

    std::string formula() const override {
        return "P(y=1|X) = 1 / (1 + exp(-z)), where z = w^T * X + b";
    }

    std::string gradient_formula() const override {
        return "∂L/∂w = (P(y=1|X) - y) * X, ∂L/∂b = P(y=1|X) - y";
    }
    
    Eigen::VectorXd get_weights() const {
        return weights_;
    }   
    Eigen::VectorXd get_bias() const {
        return bias_;
    }
    double get_learning_rate() const {
        return lr_;
    }
    int get_epochs() const {
        return epochs_;
    }
    int get_batch_size() const {
        return batch_size_;
    }
    void set_learning_rate(double lr) {
        if (lr <= 0.0)
            throw std::invalid_argument("Learning rate must be positive.");
        lr_ = lr;
    }
    void set_epochs(int epochs) {
        if (epochs <= 0)
            throw std::invalid_argument("Number of epochs must be positive.");
        epochs_ = epochs;
    }
    void set_batch_size(int batch_size) {
        if (batch_size <= 0)
            throw std::invalid_argument("Batch size must be positive.");
        batch_size_ = batch_size;
    }
    
};
