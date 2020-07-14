/**
 * \addtogroup machine_learning Machine Learning Algorithms
 * @{
 * \file
 * \brief [Adaptive Linear Neuron
 * (ADALINE)](https://en.wikipedia.org/wiki/ADALINE) implementation
 *
 * \author [Krishna Vedala](https://github.com/kvedala)
 *
 * \details
 * <a href="https://commons.wikimedia.org/wiki/File:Adaline_flow_chart.gif"><img
 * src="https://upload.wikimedia.org/wikipedia/commons/b/be/Adaline_flow_chart.gif"
 * alt="Structure of an ADALINE network. Source: Wikipedia"
 * style="width:200px; float:right;"></a>
 *
 * ADALINE is one of the first and simplest single layer artificial neural
 * network. The algorithm essentially implements a linear function
 * \f[ f\left(x_0,x_1,x_2,\ldots\right) =
 * \sum_j x_jw_j+\theta
 * \f]
 * where \f$x_j\f$ are the input features of a sample, \f$w_j\f$ are the
 * coefficients of the linear function and \f$\theta\f$ is a constant. If we
 * know the \f$w_j\f$, then for any given set of features, \f$y\f$ can be
 * computed. Computing the \f$w_j\f$ is a supervised learning algorithm wherein
 * a set of features and their corresponding outputs are given and weights are
 * computed using stochastic gradient descent method.
 */

#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>
#include <cmath>
#include <complex>
#include <ctime>
#include <iostream>
#include <stdexcept>

/** Maximum number of iterations to learn */
constexpr int MAX_ITER = 500;  // INT_MAX

/** \namespace machine_learning
 * \brief Machine learning algorithms
 */
namespace machine_learning {
class adaline {
 public:
    /**
     * Default constructor
     * \param[in] num_features number of features present
     * \param[in] eta learning rate (optional, default=0.1)
     * \param[in] convergence accuracy (optional,
     * default=\f$1\times10^{-5}\f$)
     */
    explicit adaline(int num_features, const double eta = 0.01f,
                     const double accuracy = 1e-5)
        : eta(eta), accuracy(accuracy) {
        if (eta <= 0) {
            std::cerr << "learning rate should be positive and nonzero"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        weights = std::vector<double>(
            num_features +
            1);  // additional weight is for the constant bias term

        // initialize with random weights in the range [-50, 49]
        for (double &weight : weights) weight = 1.f;
        // weights[i] = (static_cast<double>(std::rand() % 100) - 50);
    }

    /**
     * Operator to print the weights of the model
     */
    friend std::ostream &operator<<(std::ostream &out, const adaline &ada) {
        out << "<";
        for (int i = 0; i < ada.weights.size(); i++) {
            out << ada.weights[i];
            if (i < ada.weights.size() - 1) {
                out << ", ";
            }
        }
        out << ">";
        return out;
    }

    /**
     * predict the output of the model for given set of features
     * \param[in] x input vector
     * \param[out] out optional argument to return neuron output before
     * applying activation function (optional, `nullptr` to ignore) \returns
     * model prediction output
     */
    int predict(const std::vector<double> &x, double *out = nullptr) {
        if (!check_size_match(x)) {
            return 0;
        }

        double y = weights.back();  // assign bias value

        // for (int i = 0; i < x.size(); i++) y += x[i] * weights[i];
        y = std::inner_product(x.begin(), x.end(), weights.begin(), y);

        if (out != nullptr) {  // if out variable is provided
            *out = y;
        }

        return activation(y);  // quantizer: apply ADALINE threshold function
    }

    /**
     * Update the weights of the model using supervised learning for one
     * feature vector
     * \param[in] x feature vector
     * \param[in] y known output value
     * \returns correction factor
     */
    double fit(const std::vector<double> &x, const int &y) {
        if (!check_size_match(x)) {
            return 0;
        }

        /* output of the model with current weights */
        int p = predict(x);
        int prediction_error = y - p;  // error in estimation
        double correction_factor = eta * prediction_error;

        /* update each weight, the last weight is the bias term */
        for (int i = 0; i < x.size(); i++) {
            weights[i] += correction_factor * x[i];
        }
        weights[x.size()] += correction_factor;  // update bias

        return correction_factor;
    }

    /**
     * Update the weights of the model using supervised learning for an
     * array of vectors.
     * \param[in] X array of feature vector
     * \param[in] y known output value for each feature vector
     */
    template <size_t N>
    void fit(std::array<std::vector<double>, N> const &X,
             std::array<int, N> const &Y) {
        double avg_pred_error = 1.f;

        int iter = 0;
        for (iter = 0; (iter < MAX_ITER) && (avg_pred_error > accuracy);
             iter++) {
            avg_pred_error = 0.f;

            // perform fit for each sample
            for (int i = 0; i < N; i++) {
                double err = fit(X[i], Y[i]);
                avg_pred_error += std::abs(err);
            }
            avg_pred_error /= N;

            // Print updates every 200th iteration
            // if (iter % 100 == 0)
            std::cout << "\tIter " << iter << ": Training weights: " << *this
                      << "\tAvg error: " << avg_pred_error << std::endl;
        }

        if (iter < MAX_ITER) {
            std::cout << "Converged after " << iter << " iterations."
                      << std::endl;
        } else {
            std::cout << "Did not converge after " << iter << " iterations."
                      << std::endl;
        }
    }

    /** Defines activation function as Heaviside's step function.
     * \f[
     * f(x) = \begin{cases}
     * -1 & \forall x \le 0\\
     *  1 & \forall x > 0
     * \end{cases}
     * \f]
     * @param x input value to apply activation on
     * @return activation output
     */
    int activation(double x) { return x > 0 ? 1 : -1; }

 private:
    /**
     * convenient function to check if input feature vector size matches the
     * model weights size
     * \param[in] x fecture vector to check
     * \returns `true` size matches
     * \returns `false` size does not match
     */
    bool check_size_match(const std::vector<double> &x) {
        if (x.size() != (weights.size() - 1)) {
            std::cerr << __func__ << ": "
                      << "Number of features in x does not match the feature "
                         "dimension in model!"
                      << std::endl;
            return false;
        }
        return true;
    }

    const double eta;             ///< learning rate of the algorithm
    const double accuracy;        ///< model fit convergence accuracy
    std::vector<double> weights;  ///< weights of the neural network
};

}  // namespace machine_learning

using machine_learning::adaline;

/** @} */

/**
 * test function to predict points in a 2D coordinate system above the line
 * \f$x=y\f$ as +1 and others as -1.
 * Note that each point is defined by 2 values or 2 features.
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test1(double eta = 0.01) {
    adaline ada(2, eta);  // 2 features

    const int N = 10;  // number of sample points

    std::array<std::vector<double>, N> X = {
        std::vector<double>({0, 1}),   std::vector<double>({1, -2}),
        std::vector<double>({2, 3}),   std::vector<double>({3, -1}),
        std::vector<double>({4, 1}),   std::vector<double>({6, -5}),
        std::vector<double>({-7, -3}), std::vector<double>({-8, 5}),
        std::vector<double>({-9, 2}),  std::vector<double>({-10, -15})};
    std::array<int, N> y = {1,  -1, 1, -1, -1,
                            -1, 1,  1, 1,  -1};  // corresponding y-values

    std::cout << "------- Test 1 -------" << std::endl;
    std::cout << "Model before fit: " << ada << std::endl;

    ada.fit<N>(X, y);
    std::cout << "Model after fit: " << ada << std::endl;

    int predict = ada.predict({5, -3});
    std::cout << "Predict for x=(5,-3): " << predict;
    assert(predict == -1);
    std::cout << " ...passed" << std::endl;

    predict = ada.predict({5, 8});
    std::cout << "Predict for x=(5,8): " << predict;
    assert(predict == 1);
    std::cout << " ...passed" << std::endl;
}

/**
 * test function to predict points in a 2D coordinate system above the line
 * \f$x+3y=-1\f$ as +1 and others as -1.
 * Note that each point is defined by 2 values or 2 features.
 * The function will create random sample points for training and test purposes.
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test2(double eta = 0.01) {
    adaline ada(2, eta);  // 2 features

    const int N = 50;  // number of sample points

    std::array<std::vector<double>, N> X;
    std::array<int, N> Y{};  // corresponding y-values

    // generate sample points in the interval
    // [-range2/100 , (range2-1)/100]
    int range = 500;          // sample points full-range
    int range2 = range >> 1;  // sample points half-range
    for (int i = 0; i < N; i++) {
        double x0 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x1 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        X[i] = std::vector<double>({x0, x1});
        Y[i] = (x0 + 3. * x1) > -1 ? 1 : -1;
    }

    std::cout << "------- Test 2 -------" << std::endl;
    std::cout << "Model before fit: " << ada << std::endl;

    ada.fit(X, Y);
    std::cout << "Model after fit: " << ada << std::endl;

    int N_test_cases = 5;
    for (int i = 0; i < N_test_cases; i++) {
        double x0 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x1 = (static_cast<double>(std::rand() % range) - range2) / 100.f;

        int predict = ada.predict({x0, x1});

        std::cout << "Predict for x=(" << x0 << "," << x1 << "): " << predict;

        int expected_val = (x0 + 3. * x1) > -1 ? 1 : -1;
        assert(predict == expected_val);
        std::cout << " ...passed" << std::endl;
    }
}

/**
 * test function to predict points in a 3D coordinate system lying within the
 * sphere of radius 1 and centre at origin as +1 and others as -1. Note that
 * each point is defined by 3 values but we use 6 features. The function will
 * create random sample points for training and test purposes.
 * The sphere centred at origin and radius 1 is defined as:
 * \f$x^2+y^2+z^2=r^2=1\f$ and if the \f$r^2<1\f$, point lies within the sphere
 * else, outside.
 *
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test3(double eta = 0.01) {
    adaline ada(6, eta);  // 2 features

    const int N = 100;  // number of sample points

    std::array<std::vector<double>, N> X;
    std::array<int, N> Y{};  // corresponding y-values

    // generate sample points in the interval
    // [-range2/100 , (range2-1)/100]
    int range = 200;          // sample points full-range
    int range2 = range >> 1;  // sample points half-range
    for (int i = 0; i < N; i++) {
        double x0 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x1 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x2 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        X[i] = std::vector<double>({x0, x1, x2, x0 * x0, x1 * x1, x2 * x2});
        Y[i] = ((x0 * x0) + (x1 * x1) + (x2 * x2)) <= 1.f ? 1 : -1;
    }

    std::cout << "------- Test 3 -------" << std::endl;
    std::cout << "Model before fit: " << ada << std::endl;

    ada.fit(X, Y);
    std::cout << "Model after fit: " << ada << std::endl;

    int N_test_cases = 5;
    for (int i = 0; i < N_test_cases; i++) {
        double x0 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x1 = (static_cast<double>(std::rand() % range) - range2) / 100.f;
        double x2 = (static_cast<double>(std::rand() % range) - range2) / 100.f;

        int predict = ada.predict({x0, x1, x2, x0 * x0, x1 * x1, x2 * x2});

        std::cout << "Predict for x=(" << x0 << "," << x1 << "," << x2
                  << "): " << predict;

        int expected_val = ((x0 * x0) + (x1 * x1) + (x2 * x2)) <= 1.f ? 1 : -1;
        assert(predict == expected_val);
        std::cout << " ...passed" << std::endl;
    }
}
/**
 * @author tjgurwara99
 * @file
 *
 * \brief An implementation of Complex Number as Objects
 * \details A basic implementation of Complex Number field as a class with
 * operators overloaded to accommodate (mathematical) field operations.
 */
/**
 * \brief Class Complex to represent complex numbers as a field.
 */
class Complex {
    // The real value of the complex number
    double re;
    // The imaginary value of the complex number
    double im;

 public:
    /**
     * \brief Complex Constructor which initialises our complex number.
     * \details
     * Complex Constructor which initialises the complex number which takes
     * three arguments.
     * @param x If the third parameter is 'true' then this x is the absolute
     * value of the complex number, if the third parameter is 'false' then this
     * x is the real value of the complex number (optional).
     * @param y If the third parameter is 'true' then this y is the argument of
     * the complex number, if the third parameter is 'false' then this y is the
     * imaginary value of the complex number (optional).
     * @param is_polar 'false' by default. If we want to initialise our complex
     * number using polar form then set this to true, otherwise set it to false
     * to use initialiser which initialises real and imaginary values using the
     * first two parameters (optional).
     */
    explicit Complex(double x = 0.f, double y = 0.f, bool is_polar = false) {
        if (!is_polar) {
            re = x;
            im = y;
            return;
        }

        re = x * std::cos(y);
        im = x * std::sin(y);
    }

    /**
     * \brief Copy Constructor
     * @param other The other number to equate our number to.
     */
    Complex(const Complex &other) : re(other.real()), im(other.imag()) {}

    /**
     * \brief Member function to get real value of our complex number.
     * Member function (getter) to access the class' re value.
     */
    double real() const { return this->re; }

    /**
     * \brief Member function to get imaginary value of our complex number.
     * Member function (getter) to access the class' im value.
     */
    double imag() const { return this->im; }

    /**
     * \brief Member function to give the modulus of our complex number.
     * Member function to which gives the absolute value (modulus) of our
     * complex number
     * @return \f$ \sqrt{z \bar{z}} \f$ where \f$ z \f$ is our complex
     * number.
     */
    double abs() const {
        return std::sqrt(this->re * this->re + this->im * this->im);
    }

    /**
     * \brief Member function to give the argument of our complex number.
     * @return Argument of our Complex number in radians.
     */
    double arg() const { return std::atan2(this->im, this->re); }

    /**
     * \brief Operator overload of '+' on Complex class.
     * Operator overload to be able to add two complex numbers.
     * @param other The other number that is added to the current number.
     * @return result current number plus other number
     */
    Complex operator+(const Complex &other) {
        Complex result(this->re + other.re, this->im + other.im);
        return result;
    }

    /**
     * \brief Operator overload of '-' on Complex class.
     * Operator overload to be able to subtract two complex numbers.
     * @param other The other number being subtracted from the current number.
     * @return result current number subtract other number
     */
    Complex operator-(const Complex &other) {
        Complex result(this->re - other.re, this->im - other.im);
        return result;
    }

    /**
     * \brief Operator overload of '*' on Complex class.
     * Operator overload to be able to multiple two complex numbers.
     * @param other The other number to multiply the current number to.
     * @return result current number times other number.
     */
    Complex operator*(const Complex &other) {
        Complex result(this->re * other.re - this->im * other.im,
                       this->re * other.im + this->im * other.re);
        return result;
    }

    /**
     * \brief Operator overload of '~' on Complex class.
     * Operator overload of the BITWISE NOT which gives us the conjugate of our
     * complex number. NOTE: This is overloading the BITWISE operator but its
     * not a BITWISE operation in this definition.
     * @return result The conjugate of our complex number.
     */
    Complex operator~() const {
        Complex result(this->re, -(this->im));
        return result;
    }

    /**
     * \brief Operator overload of '/' on Complex class.
     * Operator overload to be able to divide two complex numbers. This function
     * would throw an exception if the other number is zero.
     * @param other The other number we divide our number by.
     * @return result Current number divided by other number.
     */
    Complex operator/(const Complex &other) {
        Complex result = *this * ~other;
        double denominator =
            other.real() * other.real() + other.imag() * other.imag();
        if (denominator != 0) {
            result = Complex(result.real() / denominator,
                             result.imag() / denominator);
            return result;
        } else {
            throw std::invalid_argument("Undefined Value");
        }
    }

    /**
     * \brief Operator overload of '=' on Complex class.
     * Operator overload to be able to copy RHS instance of Complex to LHS
     * instance of Complex
     */
    const Complex &operator=(const Complex &other) {
        this->re = other.real();
        this->im = other.imag();
        return *this;
    }
};

/**
 * \brief Operator overload of '==' on Complex class.
 * Logical Equal overload for our Complex class.
 * @param a Left hand side of our expression
 * @param b Right hand side of our expression
 * @return 'True' If real and imaginary parts of a and b are same
 * @return 'False' Otherwise.
 */
bool operator==(const Complex &a, const Complex &b) {
    return a.real() == b.real() && a.imag() == b.imag();
}

/**
 * \brief Operator overload of '<<' of ostream for Complex class.
 * Overloaded insersion operator to accommodate the printing of our complex
 * number in their standard form.
 * @param os The console stream
 * @param num The complex number.
 */
std::ostream &operator<<(std::ostream &os, const Complex &num) {
    os << "(" << num.real();
    if (num.imag() < 0) {
        os << " - " << -num.imag();
    } else {
        os << " + " << num.imag();
    }
    os << "i)";
    return os;
}

/**
 * \brief Function to get random numbers to generate our complex numbers for
 * test
 */
double get_rand() { return (std::rand() % 100 - 50) / 100.f; }

/**
 * Tests Function
 */
void tests() {
    std::srand(std::time(nullptr));
    double x1 = get_rand(), y1 = get_rand(), x2 = get_rand(), y2 = get_rand();
    Complex num1(x1, y1), num2(x2, y2);
    std::complex<double> cnum1(x1, y1), cnum2(x2, y2);
    Complex result;
    std::complex<double> expected;
    // Test for addition
    result = num1 + num2;
    expected = cnum1 + cnum2;
    assert(((void)"1 + 1i + 1 + 1i is equal to 2 + 2i but the addition doesn't "
                  "add up \n",
            (result.real() == expected.real() &&
             result.imag() == expected.imag())));
    std::cout << "First test passes." << std::endl;
    // Test for subtraction
    result = num1 - num2;
    expected = cnum1 - cnum2;
    assert(((void)"1 + 1i - 1 - 1i is equal to 0 but the program says "
                  "otherwise. \n",
            (result.real() == expected.real() &&
             result.imag() == expected.imag())));
    std::cout << "Second test passes." << std::endl;
    // Test for multiplication
    result = num1 * num2;
    expected = cnum1 * cnum2;
    assert(((void)"(1 + 1i) * (1 + 1i) is equal to 2i but the program says "
                  "otherwise. \n",
            (result.real() == expected.real() &&
             result.imag() == expected.imag())));
    std::cout << "Third test passes." << std::endl;
    // Test for division
    result = num1 / num2;
    expected = cnum1 / cnum2;
    assert(((void)"(1 + 1i) / (1 + 1i) is equal to 1 but the program says "
                  "otherwise.\n",
            (result.real() == expected.real() &&
             result.imag() == expected.imag())));
    std::cout << "Fourth test passes." << std::endl;
    // Test for conjugates
    result = ~num1;
    expected = std::conj(cnum1);
    assert(((void)"(1 + 1i) has a conjugate which is equal to (1 - 1i) but the "
                  "program says otherwise.\n",
            (result.real() == expected.real() &&
             result.imag() == expected.imag())));
    std::cout << "Fifth test passes.\n";
    // Test for Argument of our complex number
    assert(((void)"(1 + 1i) has argument PI / 4 but the program differs from "
                  "the std::complex result.\n",
            (num1.arg() == std::arg(cnum1))));
    std::cout << "Sixth test passes.\n";
    // Test for absolute value of our complex number
    assert(((void)"(1 + 1i) has absolute value sqrt(2) but the program differs "
                  "from the std::complex result. \n",
            (num1.abs() == std::abs(cnum1))));
    std::cout << "Seventh test passes.\n";
}


/** Main function */
int main(int argc, char **argv) {
    std::srand(std::time(nullptr));  // initialize random number generator

    double eta = 0.1;  // default value of eta
    if (argc == 2) {   // read eta value from commandline argument if present
        eta = strtof(argv[1], nullptr);
    }

    test1(eta);

    std::cout << "Press ENTER to continue..." << std::endl;
    std::cin.get();

    test2(eta);

    std::cout << "Press ENTER to continue..." << std::endl;
    std::cin.get();

    test3(eta);
    tests();
    return 0;
}
