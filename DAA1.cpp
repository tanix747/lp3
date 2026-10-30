#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1)
        return n;

    return fibonacci(n - 1) + fibonacci(n - 2);
}

int fibonacci(int n) {
    if (n <= 1)
        return n;

    int a = 0, b = 1;

    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }

    return b;
}

int main() {
    int n;
    cin >> n;

    cout << fibonacci(n);
    return 0;
}