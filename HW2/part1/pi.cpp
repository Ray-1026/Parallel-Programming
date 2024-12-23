#include <atomic>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "wy.hpp"
#ifndef __AVX2_AVAILABLE__
#define __AVX2_AVAILABLE__

#include "Xoshiro256Plus.h"
typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> Xoshiro256PlusAVX2;

#endif

#define ll long long
#define RANDMAX 0x7fffffff

wy::rand r;

void monte_carlo_AVX2(ll tosses, std::atomic<ll> &total_sum)
{
    unsigned int seed = r.uniform_dist(100);
    Xoshiro256PlusAVX2 rng(seed);

    ll local_sum = 0;
    __m256 x, y, sum, cmp;
    const __m256 rand_max = _mm256_set1_ps(RANDMAX), one = _mm256_set1_ps(1.0);

    for (ll i = 0; i < tosses; i += 8) {
        x = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
        x = _mm256_div_ps(x, rand_max);                         // scale to (-1, 1)

        y = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
        y = _mm256_div_ps(y, rand_max);                         // scale to (-1, 1)

        x = _mm256_mul_ps(x, x);   // x = x * x
        y = _mm256_mul_ps(y, y);   // y = y * y
        sum = _mm256_add_ps(x, y); // sum = x + y

        cmp = _mm256_cmp_ps(sum, one, _CMP_LE_OQ); // cmp = sum <= 1.0

        local_sum += _mm_popcnt_u32(_mm256_movemask_ps(cmp));
    }

    total_sum += local_sum;
}

void monte_carlo(ll tosses, std::atomic<ll> &total_sum)
{
    ll local_sum = 0;
    for (ll i = 0; i < tosses; ++i) {
        double x = r.uniform_dist(), y = r.uniform_dist();
        if (x * x + y * y <= 1.0) {
            ++local_sum;
        }
    }
    total_sum += local_sum;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <thread nums> <tosses>\n", argv[0]);
        return 1;
    }

    // monte carlo simulation
    int thread_nums = atoi(argv[1]);
    ll tosses = atoll(argv[2]);
    ll tosses_per_thread = (((tosses + thread_nums - 1) / thread_nums + 7) >> 3) << 3; // align to thread_nums and 8

    std::vector<std::thread> threads;
    std::atomic<ll> total_sum(0);

    for (int i = 0; i < thread_nums; ++i) {
        threads.emplace_back(monte_carlo_AVX2, tosses_per_thread, std::ref(total_sum));
    }

    for (std::thread &it : threads) {
        it.join();
    }

    double pi = 4.0 * total_sum / static_cast<double>(tosses);
    printf("%.10f\n", pi);

    return 0;
}