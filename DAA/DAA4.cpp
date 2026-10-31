#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> board;

bool isSafe(int row, int col) {

    // Check column
    for (int i = 0; i < row; i++)
        if (board[i][col])
            return false;

    // Check upper-left diagonal
    for (int i = row - 1, j = col - 1;
         i >= 0 && j >= 0;
         i--, j--)
        if (board[i][j])
            return false;

    // Check upper-right diagonal
    for (int i = row - 1, j = col + 1;
         i >= 0 && j < n;
         i--, j++)
        if (board[i][j])
            return false;

    return true;
}

bool solve(int row) {

    if (row == n)
    return true;

    for (int col = 0; col < n; col++) {

        if (isSafe(row, col)) {

            board[row][col] = 1;

            if (solve(row + 1))
                return true;

            board[row][col] = 0;   // Backtrack
        }
    }

    return false;
}

int main() {

    cin >> n;

    // Initialize global board
    board = vector<vector<int>>(n, vector<int>(n, 0));

    // First queen already placed
    board[0][0] = 1;

    if (solve(1)) {

        for (auto row : board) {
            for (int x : row)
                cout << x << " ";

            cout << endl;
        }

    } else {
        cout << "No solution exists";
    }

    return 0;
}