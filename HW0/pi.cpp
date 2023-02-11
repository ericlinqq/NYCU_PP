// pi.cpp
#include <iostream>
#include <random>
#include <ctime>
#include <iomanip>

int main(int argc, char *argv[]) {
    long long int number_in_circle = 0;
    long long int number_of_tosses = 10000000;
    std::default_random_engine generator(time(NULL));
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    for (long long int toss = 0; toss < number_of_tosses; toss++) {
        double x = unif(generator);
        double y = unif(generator);
        double distance_square = x * x  + y * y;
    
        if (distance_square <= 1) {
            number_in_circle++;
        }
    }

    double pi_estimate = 4 * number_in_circle / ((double) number_of_tosses);

    std::cout << std::fixed << std::setprecision(10) << pi_estimate << std::endl;
    return 0;
}