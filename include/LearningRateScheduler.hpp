#pragma once
#include <iostream>
#include <cmath>
#include <string>

class LearningRateScheduler {
public:
    virtual double getRate(int epoch) const = 0;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual std::string formula() const = 0;
    virtual ~LearningRateScheduler() = default;
};

class ConstantLearningRateScheduler : public LearningRateScheduler {
public:
    ConstantLearningRateScheduler(double rate) : rate_(rate) {}
    double getRate(int epoch) const override { return rate_; }
    std::string name() const override { return "Constant Learning Rate"; }
    std::string description() const override { return "A constant learning rate that does not change during training."; }
    std::string formula() const override { return "lr = constant_value"; }
private:
    double rate_;
};

class ExponentialDecayLearningRateScheduler : public LearningRateScheduler {
public:
    ExponentialDecayLearningRateScheduler(double init_rate, double decay_rate)
        : init_rate_(init_rate), decay_rate_(decay_rate) {}
    double getRate(int epoch) const override {
        return init_rate_ * std::exp(-decay_rate_ * epoch);
    }
    std::string name() const override { return "Exponential Decay Learning Rate"; }
    std::string description() const override { return "A learning rate that decays exponentially over time."; }
    std::string formula() const override { return "lr = lr_initial * e^(-decay_rate * epoch)"; }
private:
    double init_rate_;
    double decay_rate_;
};
