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


struct RegressionParameters
{
    double b0 {};
    double b1 {};
};

RegressionParameters LinearRegression(const Dataset& data) 
{
    RegressionParameters B {};
    double x_mean {};
    double y_mean {};
    double b1_num {};
    double b1_den {};

    auto M {data.x.size()};

    for (auto i {0uz}; i < M; ++i)
    {
        x_mean += data.x[i];
        y_mean += data.y[i];
    }

    x_mean /= M;
    y_mean /= M;

    for (auto i {0uz}; i < M; ++i)
    {
        b1_num += (data.y[i] - y_mean) * (data.x[i] - x_mean);
        b1_den += std::pow((data.x[i] - x_mean), 2.0);
    }
    B.b1 = b1_num / b1_den;
    B.b0 = y_mean - B.b1 * x_mean;

    return B;
}

double Predictor(const RegressionParameters& B, double x)
{
    return B.b0 + B.b1 * x;
}

double MSE(const RegressionParameters& B, const Dataset& data)
{
    double MSE {};
    auto M {data.x.size()};
    for (auto i {0uz}; i < M; ++i)
    {
        MSE += std::pow((data.y[i] - B.b0 - B.b1 * data.x[i]) , 2.0);
    }
    MSE /= M;

    return MSE;
}

double R2(const RegressionParameters& B, const Dataset& data)
{
    double SSE {};
    double SST {};
    double R2 {};
    double y_mean {};
    auto M {data.x.size()};

    for (auto i {0uz}; i < M; ++i)
    {
        y_mean += data.y[i];
    }
    
    y_mean /= M;

    for (auto i {0uz}; i < M; ++i)
    {
        SSE += std::pow((data.y[i] - B.b0 - B.b1 * data.x[i]) , 2.0);
        SST += std::pow((data.y[i] - y_mean) , 2.0);
    }

    R2 = 1 - SSE / SST;
    return R2;
}

int main()
{
    Dataset data(8);

    data.x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    data.y = {2.1, 4.2, 5.8, 8.3, 10.1, 11.9, 14.2, 15.7};

    RegressionParameters R {LinearRegression(data)};

    std::cout << "b0: " << R.b0 << '\n';
    std::cout << "b1: " << R.b1 << '\n';
    std::cout << "MSE: " << MSE(R, data) << '\n';
    std::cout << "R2: " << R2(R, data) << '\n';

    return 0;
}