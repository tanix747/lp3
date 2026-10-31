def f(x):
    return (x + 3) ** 2

x = 2
lr = 0.1

for i in range(20):
    grad = 2 * (x + 3)
    x = x - lr * grad

print("Minimum x =", x)
print("Minimum y =", f(x))