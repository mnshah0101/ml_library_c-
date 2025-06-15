#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>



class Loss
{
public:
    virtual double compute(const Eigen::VectorXd &y_true,
                           const Eigen::VectorXd &y_pred) const = 0;
    // gradient w.r.t. predictions
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd &y_true,
                                     const Eigen::VectorXd &y_pred) const = 0;
    virtual ~Loss() = default;
};



class MeanSquaredError : public Loss
{
    public:
    
    double compute(const Eigen::VectorXd &y_true,
                   const Eigen::VectorXd &y_pred) const override
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("y_true and y_pred must have the same size");
        }
        return (y_true - y_pred).squaredNorm() / y_true.size();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd &y_true,
                             const Eigen::VectorXd &y_pred) const override
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("y_true and y_pred must have the same size");
        }
        return 2 * (y_pred - y_true) / y_true.size();
    }


    std::string name() const
    {
        return "Mean Squared Error";
    }
    std::string description() const
    {
        return "Mean Squared Error (MSE) is a common loss function for regression tasks. It measures the average of the squares of the errors, that is, the average squared difference between the estimated values and the actual value.";
    }
    std::string formula() const
    {
        return "MSE = (1/n) * Σ(y_true - y_pred)^2";
    }
    std::string gradient_formula() const
    {
        return "∂MSE/∂y_pred = (2/n) * (y_pred - y_true)";
    }

};

class CrossEntropy : public Loss
{  
    public:
    double compute(const Eigen::VectorXd &y_true,
                   const Eigen::VectorXd &y_pred) const override
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("y_true and y_pred must have the same size");
        }
        return - (y_true.array() * y_pred.array().log()).sum() / y_true.size();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd &y_true,
                             const Eigen::VectorXd &y_pred) const override
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("y_true and y_pred must have the same size");
        }
        return - (y_true.array() / y_pred.array()) / y_true.size();
    }
    std::string name() const
    {
        return "Cross Entropy";
    }
    std::string description() const
    {
        return "Cross Entropy is a loss function commonly used in classification tasks. It measures the dissimilarity between two probability distributions, typically the true distribution and the predicted distribution.";
    }
    std::string formula() const
    {
        return "Cross Entropy = - (1/n) * Σ(y_true * log(y_pred))";
    }
    std::string gradient_formula() const
    {
        return "∂Cross Entropy/∂y_pred = - (1/n) * (y_true / y_pred)";
    }
};

