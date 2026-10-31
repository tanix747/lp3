#include <bits/stdc++.h>
using namespace std;

// ---------------- DETERMINISTIC QUICK SORT ----------------

int partition1(vector<int>& a, int low, int high) {
    int pivot = a[high];       // Last element as pivot
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

void quickSort1(vector<int>& a, int low, int high) {
    if (low >= high)
        return;

    int p = partition1(a, low, high);

    quickSort1(a, low, p - 1);
    quickSort1(a, p + 1, high);
}


// ---------------- RANDOMIZED QUICK SORT ----------------

int partition2(vector<int>& a, int low, int high) {
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

void quickSort2(vector<int>& a, int low, int high) {
    if (low >= high)
        return;

    // Choose random pivot
    int randomIndex = low + rand() % (high - low + 1);
    swap(a[randomIndex], a[high]);

    int p = partition2(a, low, high);

    quickSort2(a, low, p - 1);
    quickSort2(a, p + 1, high);
}


// ---------------- MAIN ----------------

int main() {
    srand(time(0));

    int n;
    cin >> n;

    vector<int> a(n);

    for (int &x : a)
        cin >> x;

    vector<int> a1 = a;
    vector<int> a2 = a;

    quickSort1(a1, 0, n - 1);
    quickSort2(a2, 0, n - 1);

    cout << "Deterministic: ";
    for (int x : a1)
        cout << x << " ";

    cout << "\nRandomized: ";
    for (int x : a2)
        cout << x << " ";

    return 0;
}