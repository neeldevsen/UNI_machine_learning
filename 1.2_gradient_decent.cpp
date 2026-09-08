#include <iostream>
#include <vector>
#include <cmath>

struct Dataset
{
    std::vector<double> x {};
    std::vector<double> y {};

    Dataset(int M)
    : x(M), y(M)
    {}
};

// gradient decent struct

struct GradientDecentParameters
{
    double th0 {};
    double th1 {}; 

    GradientDecentParameters(double x, double y)
    : th0 {x}, th1 {y}
    {;}
};

// gradient struct 

struct Gradient
{
    double dth0 {};
    double dth1 {};

    Gradient (const GradientDecentParameters& GDparams, const Dataset& data, const double learning_rate)
    : dth0 {0}, dth1 {0}
    {
        auto M {data.x.size()};
        for (auto i {0uz}; i < M; ++i)
        {
            dth0 += 2 * (GDparams.th0 + GDparams.th1 * data.x[i] - data.y[i]);
            dth1 += 2 * (data.x[i]) * (GDparams.th0 + GDparams.th1 * data.x[i] - data.y[i]);
        }

        dth0 *= learning_rate / M;
        dth1 *= learning_rate / M;
    }
};

// operator overloaded functions

GradientDecentParameters operator+(const GradientDecentParameters& GDParams, const Gradient& GD)
    {
        return {GDParams.th0 + GD.dth0, GDParams.th1 + GD.dth1};
    }


GradientDecentParameters operator-(const GradientDecentParameters& GDParams, const Gradient& GD)
    {
        return {GDParams.th0 - GD.dth0, GDParams.th1 - GD.dth1};
    }


// gradient decent function here

GradientDecentParameters gradientDecent(const Dataset& data, GradientDecentParameters& initial, double learning_rate, int N)
{
    GradientDecentParameters GDParams {initial};
    for (int j {0}; j < N; ++j)
    {
       Gradient grad_adjusted {GDParams, data, learning_rate};
       GDParams = GDParams - grad_adjusted;
    }
    return GDParams;
}

int main()
{
    Dataset data(8);
    double learning_rate {0.01};
    int N {1000};

    data.x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    data.y = {2.1, 4.2, 5.8, 8.3, 10.1, 11.9, 14.2, 15.7};

    GradientDecentParameters Th {0.3,0.5};
    Th = gradientDecent(data, Th, learning_rate, N);

    std::cout << "th0: " << Th.th0 << '\n';
    std::cout << "th1: " << Th.th1 << '\n';

    return 0;
}