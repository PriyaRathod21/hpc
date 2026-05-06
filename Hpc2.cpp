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

// BEGINNING OF CODE
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

void printArray(const vector<int>& arr) {
    for (int num : arr)
        cout << num << " ";
    cout << endl;
}

// Bubble Sort

// Sequential bubble sort.
// Sorts the array using bubble sort by repeatedly swapping adjacent elements.
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// Parallel bubble sort using odd-even transposition.
// Standard bubble sort cannot be parallelized directly: thread on index j
// and thread on index j+1 would both touch arr[j+1] simultaneously (data race).
// Odd-even transposition alternates between two phases each pass:
//   Phase 0 (even): compare pairs (0,1), (2,3), (4,5), ...
//   Phase 1 (odd):  compare pairs (1,2), (3,4), (5,6), ...
// Within each phase every pair is independent, so threads never share elements.
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        // i % 2 selects even phase (0) or odd phase (1).
        // The starting index of the first pair in each phase matches i % 2.
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// Merge Sort

// Merges two sorted halves arr[left..mid] and arr[mid+1..right] in place.
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Parallel merge sort using OpenMP tasks.
// "#pragma omp parallel sections" inside a recursive function would spawn a
// new thread team at every level of recursion, hundreds of thousands of teams
// for a large array, causing enormous overhead and likely a crash.
// Tasks are lighter: the runtime schedules them across an existing thread pool.
// The depth cutoff switches to sequential below a threshold to avoid spawning
// tasks so small that the overhead exceeds the work itself.
void parallelMergeSortHelper(vector<int>& arr, int left, int right, int depth) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;

    if (depth <= 0) {
        // Below the cutoff the subarray is small enough that sequential is faster.
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
    } else {
        #pragma omp task
        parallelMergeSortHelper(arr, left, mid, depth - 1);

        #pragma omp task
        parallelMergeSortHelper(arr, mid + 1, right, depth - 1);

        // Wait for both tasks to finish before merging.
        #pragma omp taskwait
    }

    merge(arr, left, mid, right);
}

void parallelMergeSort(vector<int>& arr, int left, int right) {
    // The single directive creates one thread team for the entire sort.
    // All recursive tasks share this pool instead of creating new teams.
    #pragma omp parallel
    {
        // single ensures only one thread kicks off the root task;
        // the rest wait and pick up the child tasks as they are created.
        #pragma omp single
        parallelMergeSortHelper(arr, left, right, 4); // depth 4 → up to 16 parallel tasks
    }
}

// Main function

int main() {
    int n = 10000; // Adjust this to specify the number of elements.
    vector<int> arr(n);

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    double start, end;
    double time_seq_bubble, time_par_bubble;
    double time_seq_merge, time_par_merge;

    // --- Sequential Bubble Sort ---
    vector<int> seqArr = arr;
    start = omp_get_wtime();
    sequentialBubbleSort(seqArr);
    end = omp_get_wtime();
    time_seq_bubble = end - start;
    cout << "Sequential Bubble Sort time: " << time_seq_bubble << " seconds" << endl;

    // --- Parallel Bubble Sort ---
    vector<int> parArr = arr;
    start = omp_get_wtime();
    omp_set_num_threads(2);
    parallelBubbleSort(parArr);
    end = omp_get_wtime();
    time_par_bubble = end - start;
    cout << "Parallel Bubble Sort time: " << time_par_bubble << " seconds" << endl;

    cout << "Bubble Sort Speedup (Sequential / Parallel) = " << (time_seq_bubble / time_par_bubble) << "x" << endl;

    // --- Sequential Merge Sort ---
    seqArr = arr;
    start = omp_get_wtime();
    sequentialMergeSort(seqArr, 0, n - 1);
    end = omp_get_wtime();
    time_seq_merge = end - start;
    cout << "\nSequential Merge Sort time: " << time_seq_merge << " seconds" << endl;

    // --- Parallel Merge Sort ---
    parArr = arr;
    start = omp_get_wtime();
    omp_set_num_threads(4);
    parallelMergeSort(parArr, 0, n - 1);
    end = omp_get_wtime();
    time_par_merge = end - start;
    cout << "Parallel Merge Sort time: " << time_par_merge << " seconds" << endl;

    cout << "Merge Sort Speedup (Sequential / Parallel) = " << (time_seq_merge / time_par_merge) << "x" << endl;

    return 0;
}
