#pragma once
#include <iostream>
#include <string>
#include <Eigen/Dense>

// Forward declaration
class Dataset;

class Model {
    public: 
        virtual void fit(const Dataset &train) = 0;
        virtual Eigen::VectorXd predict(const Eigen::MatrixXd &X) const = 0;
        virtual void update_parameters(Eigen::VectorXd gradients, double rate) = 0;
        virtual std::string name() const = 0;
        virtual std::string description() const = 0;
        virtual std::string formula() const = 0;
        virtual std::string gradient_formula() const = 0;
        virtual ~Model() = default;
};
