#pragma once
#include <Eigen/Dense>
#include <random>
#include <limits>
#include <string>

class KMeans
{
private:
    int k_;
    int max_iters_;
    Eigen::MatrixXd centroids_; // [k x n_features]
    std::mt19937 rng_; // Random number generator

public:
    KMeans(int k = 3, int max_iters = 100) : k_(k), max_iters_(max_iters), rng_(std::random_device{}()) {
        if (k <= 0) {
            throw std::invalid_argument("Number of clusters k must be positive");
        }
        if (max_iters <= 0) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
    }
    void fit(const Eigen::MatrixXd &X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Input matrix X cannot be empty");
        }
        if (X.rows() < k_) {
            throw std::invalid_argument("Number of samples must be greater than number of clusters");
        }

        // Initialize centroids randomly
        centroids_ = Eigen::MatrixXd::Zero(k_, X.cols());
        std::uniform_int_distribution<int> dist(0, X.rows() - 1);
        for (int i = 0; i < k_; i++) {
            centroids_.row(i) = X.row(dist(rng_));
        }

        // Main K-means loop
        for (int iter = 0; iter < max_iters_; iter++) {
            // Assign points to nearest centroid
            Eigen::VectorXi labels = assign_points(X);
            
            // Update centroids
            Eigen::MatrixXd new_centroids = update_centroids(X, labels);
            
            // Check for convergence
            if ((new_centroids - centroids_).norm() < 1e-6) {
                break;
            }
            centroids_ = new_centroids;
        }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Input matrix X cannot be empty");
        }
        if (X.cols() != centroids_.cols()) {
            throw std::invalid_argument("Input dimensions do not match training data");
        }
        return assign_points(X);
    }

    Eigen::MatrixXd update_centroids(const Eigen::MatrixXd &X, const Eigen::VectorXi &labels) const {
        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k_, X.cols());
        Eigen::VectorXi cluster_sizes = Eigen::VectorXi::Zero(k_);

        // Sum up points in each cluster
        for (int i = 0; i < X.rows(); i++) {
            int cluster = labels(i);
            new_centroids.row(cluster) += X.row(i);
            cluster_sizes(cluster)++;
        }

        // Compute means
        for (int i = 0; i < k_; i++) {
            if (cluster_sizes(i) > 0) {
                new_centroids.row(i) /= cluster_sizes(i);
            }
        }

        return new_centroids;
    }

    Eigen::VectorXi assign_points(const Eigen::MatrixXd &X) const {
        Eigen::VectorXi labels(X.rows());
        
        for (int i = 0; i < X.rows(); i++) {
            double min_dist = std::numeric_limits<double>::infinity();
            int min_idx = 0;
            
            for (int j = 0; j < k_; j++) {
                double dist = (X.row(i) - centroids_.row(j)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            labels(i) = min_idx;
        }
        
        return labels;
    }

    // Getter for centroids
    Eigen::MatrixXd get_centroids() const {
        return centroids_;
    }

    // Getter for number of clusters
    int get_k() const {
        return k_;
    }

    // Getter for maximum iterations
    int get_max_iters() const {
        return max_iters_;
    }

    std::string name() const {
        return "KMeans";
    }
    std::string description() const {
        return "KMeans is a clustering algorithm that partitions the data into k clusters.";
    }
    std::string formula() const {
        return "argmin_S sum_{i=1}^k sum_{x in S_i} ||x - mu_i||^2";
    }
    std::string gradient_formula() const {
        return "Not applicable - KMeans is not a gradient-based algorithm";
    }
    ~KMeans() = default;
    
    
    
};