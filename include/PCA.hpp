#pragma once
#include "model.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include "LearningRateScheduler.hpp"
#include "dataset.hpp"

class PCA : public Model {
    private:
        Eigen::MatrixXd _components;
        Eigen::VectorXd _explained_variance;
        Eigen::VectorXd _explained_variance_ratio;
        int _n_components;

    public:
        PCA(int n_components = 2) : _n_components(n_components) {
            if (n_components <= 0) {
                throw std::invalid_argument("Number of components must be positive");
            }
        }

        void fit(const Dataset &train) override {
            Eigen::MatrixXd X = train.getX();
            if (X.rows() == 0 || X.cols() == 0) {
                throw std::invalid_argument("Input matrix cannot be empty");
            }
            if (_n_components > X.cols()) {
                throw std::invalid_argument("Number of components cannot be greater than number of features");
            }

            // Center the data
            Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
            
            // Compute covariance matrix
            Eigen::MatrixXd cov = centered.transpose() * centered / (X.rows() - 1);
            
            // Compute eigendecomposition
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
            
            // Sort eigenvalues and eigenvectors in descending order
            Eigen::VectorXd eigenvalues = eig.eigenvalues().reverse();
            Eigen::MatrixXd eigenvectors = eig.eigenvectors().rowwise().reverse();
            
            // Store results
            _components = eigenvectors.leftCols(_n_components);
            _explained_variance = eigenvalues.head(_n_components);
            _explained_variance_ratio = _explained_variance.array() / eigenvalues.sum();
        }

        Eigen::VectorXd predict(const Eigen::MatrixXd &X) const override {
            // For PCA, predict is the same as transform
            return transform(X).rowwise().norm();
        }

        void update_parameters(Eigen::VectorXd gradients, double rate) override {
            throw std::logic_error("PCA does not support parameter updates");
        }

        Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
            if (X.rows() == 0 || X.cols() == 0) {
                throw std::invalid_argument("Input matrix cannot be empty");
            }
            if (X.cols() != _components.rows()) {
                throw std::invalid_argument("Input dimensions do not match training data");
            }
            return X * _components;
        }

        Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd &X) const {
            if (X.rows() == 0 || X.cols() == 0) {
                throw std::invalid_argument("Input matrix cannot be empty");
            }
            if (X.cols() != _components.cols()) {
                throw std::invalid_argument("Input dimensions do not match transformed data");
            }
            return X * _components.transpose();
        }

        Eigen::MatrixXd get_components() const {
            return _components;
        }

        Eigen::VectorXd get_explained_variance() const {
            return _explained_variance;
        }

        Eigen::VectorXd get_explained_variance_ratio() const {
            return _explained_variance_ratio;
        }

        std::string name() const override {
            return "PCA";
        }

        std::string description() const override {
            return "PCA is a dimensionality reduction technique that reduces the number of features in a dataset by projecting the data onto a lower-dimensional space.";
        }

        std::string formula() const override {
            return "X' = X * W";
        }

        std::string gradient_formula() const override {
            return "Not applicable - PCA is not a gradient-based algorithm";
        }   

        ~PCA() override = default;
};