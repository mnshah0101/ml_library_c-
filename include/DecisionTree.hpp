#pragma once
#include "model.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include "LearningRateScheduler.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

class TreeNode
{
    private:
        int _feature;
        double _threshold;
        TreeNode* _left;
        TreeNode* _right;
        double _value;
    
        
}


class DecisionTree : public Model
{
    private:
        TreeNode* _root;
        int _max_depth;

        double _gini_impurity(const Eigen::VectorXd& y) const{
            Eigen::VectorXd unique, counts;
            Eigen::MatrixXd::Index index;
            unique = y.col(0).uniquecounts(y.col(0), unique, counts);
            double impurity = 0.0;
            for (int i = 0; i < unique.size(); i++)
            {   
                double p = counts(i) / y.size();
                impurity -= p * p;
            }
            return impurity;
        }   

        double _entropy(const Eigen::VectorXd& y) const{
            Eigen::VectorXd unique, counts;
            Eigen::MatrixXd::Index index;
            unique = y.col(0).uniquecounts(y.col(0), unique, counts);
            double entropy = 0.0;
            for (int i = 0; i < unique.size(); i++)
            {
                double p = counts(i) / y.size();
                entropy -= p * log(p);
            }
            return entropy;
        }

        
}
