#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <loss.hpp>
#include <LearningRateScheduler.hpp>
#include <dataset.hpp>
#include <model.hpp>

class Optimizer {
public:
    virtual void optimize(Model &model, const Dataset &dataset, const Loss &loss, const LearningRateScheduler &scheduler, int epochs, int batchSize) = 0;
    virtual ~Optimizer() = default;
};

class GradientDescent : public Optimizer {
private:
    double n0_;
    bool shuffle_batches_;
public:
    GradientDescent(double n0 = 0.01, bool shuffle_batches = true)
        : n0_(n0), shuffle_batches_(shuffle_batches) {

            // Validate learning rate
            if (n0 <= 0.0) {
                throw std::invalid_argument("Learning rate must be positive.");
            }
            // Validate shuffle_batches
            if (!shuffle_batches) {
                std::cout << "Warning: Shuffle batches is set to false. This may lead to suboptimal convergence." << std::endl;
            }  

            //use gradient function



        }
    void optimize(Model &model, const Dataset &dataset, const Loss &loss, const LearningRateScheduler &scheduler, int epochs, int batchSize) override {
        if (epochs <= 0) {
            throw std::invalid_argument("Number of epochs must be positive.");
        }
        if (batchSize <= 0 || batchSize > dataset.getNumRows()) {
            throw std::invalid_argument("Batch size must be positive and less than or equal to the number of samples.");
        }

        Eigen::MatrixXd X = dataset.getX();
        Eigen::VectorXd y = dataset.getY();
        int num_samples = X.rows();
        int num_features = X.cols();

        if (num_samples == 0 || num_features == 0) {
            throw std::runtime_error("Dataset is empty.");
        }

        double epoch_loss = 0.0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            if (shuffle_batches_) {
                // Get shuffled dataset and update X and y
                Dataset shuffled = dataset.shuffle(epoch);
                X = shuffled.getX();
                y = shuffled.getY();
            }

            epoch_loss = 0.0;
            for (int start = 0; start < num_samples; start += batchSize) {
                int end = std::min(start + batchSize, num_samples);
                Eigen::MatrixXd X_batch = X.middleRows(start, end - start);
                Eigen::VectorXd y_batch = y.segment(start, end - start);

                // Make predictions
                Eigen::VectorXd y_pred = model.predict(X_batch);

                // Compute loss and gradient
                double batch_loss = loss.compute(y_batch, y_pred);
                epoch_loss += batch_loss * (end - start) / num_samples;  // Weight by batch size
                
                // Compute gradients with respect to predictions
                Eigen::VectorXd pred_gradients = loss.gradient(y_batch, y_pred);
                
                // Check for nan values in predictions and gradients
                if (y_pred.hasNaN() || pred_gradients.hasNaN()) {
                    std::cerr << "Warning: NaN detected in predictions or gradients. Skipping batch." << std::endl;
                    continue;
                }
                
                // Compute gradients with respect to weights and bias
                Eigen::VectorXd weight_gradients = X_batch.transpose() * pred_gradients;
                double bias_gradient = pred_gradients.sum();
                
                // Gradient clipping to prevent exploding gradients
                double max_grad_norm = 1.0;
                double grad_norm = weight_gradients.norm();
                if (grad_norm > max_grad_norm) {
                    weight_gradients *= max_grad_norm / grad_norm;
                }
                if (std::abs(bias_gradient) > max_grad_norm) {
                    bias_gradient = (bias_gradient > 0 ? 1.0 : -1.0) * max_grad_norm;
                }
                
                // Combine gradients
                Eigen::VectorXd combined_gradients(weight_gradients.size() + 1);
                combined_gradients << weight_gradients, bias_gradient;

                // Get learning rate and check for numerical stability
                double learning_rate = scheduler.getRate(epoch);
                if (learning_rate <= 0 || std::isnan(learning_rate)) {
                    std::cerr << "Warning: Invalid learning rate detected. Using default value." << std::endl;
                    learning_rate = 0.001;
                }

                // Update model parameters
                model.update_parameters(combined_gradients, learning_rate);
            }

            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ": Loss = " << epoch_loss << std::endl;
        }
    }
    std::string name() const;
    std::string description() const;
    std::string formula() const;
    std::string gradient_formula() const;
    ~GradientDescent() override = default;
};
