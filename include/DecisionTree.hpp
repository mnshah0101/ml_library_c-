#pragma once
#include "model.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <vector>
#include "LearningRateScheduler.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

class TreeNode {
    private:
        int _feature;
        double _threshold;
        TreeNode* _left;
        TreeNode* _right;
        double _value;
        bool _is_leaf;
    
    public:
        // Constructor for leaf nodes
        TreeNode(double value) : _feature(-1), _threshold(0), _left(nullptr), _right(nullptr), 
                               _value(value), _is_leaf(true) {}
        
        // Constructor for internal nodes
        TreeNode(int feature, double threshold, TreeNode* left, TreeNode* right) 
            : _feature(feature), _threshold(threshold), _left(left), _right(right), 
              _value(0), _is_leaf(false) {}
        
        bool is_leaf() const { return _is_leaf; }
        int get_feature() const { return _feature; }
        double get_threshold() const { return _threshold; }
        TreeNode* get_left() const { return _left; }
        TreeNode* get_right() const { return _right; }
        double get_value() const { return _value; }
        
        ~TreeNode() {
            delete _left;
            delete _right;
        }
};

class DecisionTree : public Model {
    private:
        TreeNode* _root;
        int _max_depth;

        double _gini_impurity(const Eigen::VectorXd& y) const {
            // Count unique values and their frequencies
            std::map<double, int> counts;
            for(int i = 0; i < y.size(); i++) {
                counts[y(i)]++;
            }
            
            double impurity = 0.0;
            for(const auto& pair : counts) {
                double p = static_cast<double>(pair.second) / y.size();
                impurity -= p * p;
            }
            return impurity;
        }   

        double _entropy(const Eigen::VectorXd& y) const {
            // Count unique values and their frequencies
            std::map<double, int> counts;
            for(int i = 0; i < y.size(); i++) {
                counts[y(i)]++;
            }
            
            double entropy = 0.0;
            for(const auto& pair : counts) {
                double p = static_cast<double>(pair.second) / y.size();
                entropy -= p * std::log(p);
            }
            return entropy;
        }

        double _information_gain(const Eigen::VectorXd& y, const Eigen::VectorXd& y1, const Eigen::VectorXd& y2) const {
            double p = static_cast<double>(y1.size()) / y.size();
            double entropy = _entropy(y);
            double entropy1 = _entropy(y1);
            double entropy2 = _entropy(y2);
            return entropy - p * entropy1 - (1 - p) * entropy2;
        }

        TreeNode* _build_tree(const Dataset& dataset, int depth) {
            Eigen::VectorXd y = dataset.getY();
            if (y.size() == 0) return nullptr;
            
            if (depth >= _max_depth || _gini_impurity(y) == 0) {
                return new TreeNode(y.mean());
            }
            
            int best_feature = -1;
            double best_threshold = -1;
            double best_gain = -1;
            
            for (int i = 0; i < dataset.getX().cols(); i++) {
                for (int j = 0; j < dataset.getX().rows(); j++) {
                    double threshold = dataset.getX()(j, i);
                    
                    // Count samples for each split
                    int left_count = 0, right_count = 0;
                    for(int k = 0; k < y.size(); k++) {
                        if(dataset.getX()(k, i) < threshold) {
                            left_count++;
                        } else {
                            right_count++;
                        }
                    }
                    
                    // Pre-allocate vectors
                    Eigen::VectorXd y1(left_count);
                    Eigen::VectorXd y2(right_count);
                    
                    // Fill the vectors
                    int left_idx = 0, right_idx = 0;
                    for(int k = 0; k < y.size(); k++) {
                        if(dataset.getX()(k, i) < threshold) {
                            y1(left_idx++) = y(k);
                        } else {
                            y2(right_idx++) = y(k);
                        }
                    }
                    
                    double gain = _information_gain(y, y1, y2);
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feature = i;
                        best_threshold = threshold;
                    }
                }
            }
            
            if (best_feature == -1) {
                return new TreeNode(y.mean());
            }
            
            // Count samples for final split
            int left_count = 0, right_count = 0;
            for(int i = 0; i < dataset.getX().rows(); i++) {
                if(dataset.getX()(i, best_feature) < best_threshold) {
                    left_count++;
                } else {
                    right_count++;
                }
            }
            
            // Pre-allocate matrices and vectors
            Eigen::MatrixXd X1(left_count, dataset.getX().cols());
            Eigen::MatrixXd X2(right_count, dataset.getX().cols());
            Eigen::VectorXd y1(left_count);
            Eigen::VectorXd y2(right_count);
            
            // Fill the matrices and vectors
            int left_idx = 0, right_idx = 0;
            for(int i = 0; i < dataset.getX().rows(); i++) {
                if(dataset.getX()(i, best_feature) < best_threshold) {
                    X1.row(left_idx) = dataset.getX().row(i);
                    y1(left_idx++) = y(i);
                } else {
                    X2.row(right_idx) = dataset.getX().row(i);
                    y2(right_idx++) = y(i);
                }
            }
            
            TreeNode* left = _build_tree(Dataset(X1, y1), depth + 1);
            TreeNode* right = _build_tree(Dataset(X2, y2), depth + 1);
            return new TreeNode(best_feature, best_threshold, left, right);
        }

        double _predict(TreeNode* node, const Eigen::RowVectorXd& x) const {
            if (node == nullptr) return 0;
            if (node->is_leaf()) return node->get_value();
            if (x(node->get_feature()) < node->get_threshold()) {
                return _predict(node->get_left(), x);
            }
            return _predict(node->get_right(), x);
        }

    public:
        DecisionTree(int max_depth = 5) : _root(nullptr), _max_depth(max_depth) {}
        
        void fit(const Dataset& train) override {
            _root = _build_tree(train, 0);
        }
        
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override {
            Eigen::VectorXd predictions(X.rows());
            for(int i = 0; i < X.rows(); i++) {
                predictions(i) = _predict(_root, X.row(i));
            }
            return predictions;
        }
        
        void update_parameters(Eigen::VectorXd gradients, double rate) override {
            throw std::logic_error("DecisionTree does not support parameter updates.");
        }
        
        std::string name() const override {
            return "Decision Tree";
        }
        
        std::string description() const override {
            return "A decision tree is a non-parametric model that can be used for both classification and regression.";
        }
        
        std::string formula() const override {
            return "f(x) = sum(alpha_i * I(x in R_i))";
        }
        
        std::string gradient_formula() const override {
            return "None";
        }
        
        ~DecisionTree() override {
            delete _root;
        }
};
