// pi.cpp
#include <iostream>
#include <iomanip>
#include <pthread.h>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <atomic>

void* calculate_in_circle(void*);
/* Global variables*/
unsigned long thread_count;

std::atomic<unsigned long long> total_in_circle(0);
unsigned long long number_of_tosses;

int main(int argc, char *argv[]) {
    unsigned long thread;
    pthread_t* thread_handles;
    
    /* Get number of thread from command line */
    thread_count = strtol(argv[1], NULL, 10);
    
    thread_handles = new pthread_t [thread_count];    
    
    number_of_tosses = strtoll(argv[2], NULL, 10);

    unsigned long long num_per_thread = number_of_tosses / thread_count;

    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, calculate_in_circle, (void*)num_per_thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    

    double pi_estimate = 4 * total_in_circle / (double) number_of_tosses;

    delete [] thread_handles;

    std::cout << std::fixed << std::setprecision(10) << pi_estimate << std::endl;

    return 0;
}

void* calculate_in_circle(void* num_per_thread) {
    thread_local unsigned int myFile = open("/dev/urandom", O_RDONLY);
    close(myFile);
    unsigned long long num = (unsigned long long) num_per_thread;
    unsigned long long in_circle = 0;
    unsigned long long i;
    
    double x, y, dist;
    // const __m256 MAX = _mm256_set1_ps(RAND_MAX);
    // const __m256 ONES = _mm256_set1_ps(1.0f);

    // __m256 rand, x, y, dist, mask; 

    for (i = 0; i < num; i++) {
        x = (double) rand_r(&myFile) / RAND_MAX;
        y = (double) rand_r(&myFile) / RAND_MAX;
        dist = x*x + y*y;
        if (dist <= 1) in_circle++;
    }

    // for (i = 0; i < num; i += 8) {
    //     rand = _mm256_set_ps(rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed));
    //     x = _mm256_div_ps(rand, MAX);
        
    //     rand = _mm256_set_ps(rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed), rand_r(&seed));
    //     y = _mm256_div_ps(rand, MAX);
        
    //     dist = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
        
    //     mask = _mm256_cmp_ps(dist, ONES, _CMP_LT_OQ);
    //     in_circle += _mm_popcnt_u32(_mm256_movemask_ps(mask)); 
    // }
    
    total_in_circle += in_circle;

    pthread_exit(NULL);
}