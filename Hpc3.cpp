/*
 * EXECUTION INSTRUCTIONS (Debian-based distributions):
 *
 * i) Install g++ with OpenMP support:
 *   sudo apt update
 *   sudo apt install g++
 *
 * ii) Compile:
 *   g++ -fopenmp Code-2.cpp -o Code-2
 *
 * iii) Execute:
 *   ./Code-2
 * 
 * repository on KSKA Git: https://git.kska.io/sppu-be-comp-content/HighPerformanceComputing
 **/

 #include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>

using namespace std;

int main(){
    
    int n = 1000000;
    vector<int> nums(n);
    for(int i=0;i<n;i++){
        nums[i] = rand() % 10000;
    }
    cout<<"Input: "<<n<<endl;
    
    long long sums,sump;
    int mins,maxs,minp,maxp;
    double avgs,avgp;
    double start,end,times,timep;
    
    mins, maxs = nums[0];
    sums = 0;
    start = omp_get_wtime();
    for(int i=0;i<n;i++){
        if(nums[i]<mins)
            mins = nums[i];
        if(nums[i]>maxs)
            maxs = nums[i];
        sums+=nums[i];
    }
    end = omp_get_wtime();
    avgs = (double) sums/n;
    times = end-start;
    cout<<"Time sequential: "<<times<<endl;
    
    minp, maxp = nums[0];
    sump = 0;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(min: minp) reduction(max: maxp) reduction(+:sump)
    for(int i=0;i<n;i++){
        if(nums[i]<minp)
            minp = nums[i];
        if(nums[i]>maxp)
            maxp = nums[i];
        sump+=nums[i];
    }
    end = omp_get_wtime();
    avgp = (double) sump/n;
    timep = end-start;
    cout<<"Time parallel: "<<timep<<endl;
    
    // --- Output ---
    cout << "--- Sequential Computation ---" << endl;
    cout << "Minimum  : " << mins << endl;
    cout << "Maximum  : " << maxs << endl;
    cout << "Sum      : " << sums << endl;
    cout << "Average  : " << avgs << endl;
    cout << "Time     : " << times << " seconds" << endl;

    cout << "\n--- Parallel Computation ---" << endl;
    cout << "Minimum  : " << minp << endl;
    cout << "Maximum  : " << maxp << endl;
    cout << "Sum      : " << sump << endl;
    cout << "Average  : " << avgp << endl;
    cout << "Time     : " << timep << " seconds" << endl;

    cout << "\nSpeedup (Sequential / Parallel) = " << (times / timep) << "x" << endl;

    return 0;
}