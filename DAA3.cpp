#include <bits/stdc++.h>
using namespace std;

int knapsack(int i, int W, vector<int>& wt, vector<int>& val,
             vector<vector<int>>& dp) {

    if (i == 0 || W == 0)
        return 0;

    if (dp[i][W] != -1)
        return dp[i][W];

    if (wt[i - 1] > W)
        return dp[i][W] = knapsack(i - 1, W, wt, val, dp);

    return dp[i][W] = max(
        val[i - 1] + knapsack(i - 1, W - wt[i - 1], wt, val, dp),
        knapsack(i - 1, W, wt, val, dp)
    );
}

int main() {
    int n, W;

    cout << "Enter number of items: ";
    cin >> n;

    vector<int> wt(n), val(n);

    cout << "Enter weights: ";
    for (int i = 0; i < n; i++)
        cin >> wt[i];

    cout << "Enter values: ";
    for (int i = 0; i < n; i++)
        cin >> val[i];

    cout << "Enter capacity: ";
    cin >> W;

    vector<vector<int>> dp(n + 1, vector<int>(W + 1, -1));

    cout << "Maximum value = "
         << knapsack(n, W, wt, val, dp) << endl;

    return 0;
}