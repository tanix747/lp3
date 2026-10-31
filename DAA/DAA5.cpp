#include <bits/stdc++.h>
using namespace std;

int partition(vector<int>& a, int low, int high) {
    int pivot = a[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (a[j] <= pivot) {
            i++;
            swap(a[i], a[j]);
        }
    }

    swap(a[i + 1], a[high]);
    return i + 1;
}

void quickSort(vector<int>& a, int low, int high, bool randomized) {
    if (low >= high)
        return;

    if (randomized) {
        int pivotIndex = low + rand() % (high - low + 1);
        swap(a[pivotIndex], a[high]);
    }

    int p = partition(a, low, high);

    quickSort(a, low, p - 1, randomized);
    quickSort(a, p + 1, high, randomized);
}

int main() {
    srand(time(0));

    int n;
    cin >> n;

    vector<int> a(n);
    for (int &x : a)
        cin >> x;

    // Deterministic Quick Sort
    vector<int> deterministic = a;
    quickSort(deterministic, 0, n - 1, false);

    cout << "Deterministic Quick Sort: ";
    for (int x : deterministic)
        cout << x << " ";

    // Randomized Quick Sort
    vector<int> randomized = a;
    quickSort(randomized, 0, n - 1, true);

    cout << "\nRandomized Quick Sort: ";
    for (int x : randomized)
        cout << x << " ";

    return 0;
}