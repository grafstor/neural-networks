# pure python nn

'''
    author: grafstor
    date: 13.06.20
'''

__version__ = "1.0"

E = 2.71828182846

def dot(a, b):
    main = []
    for arr in a:
        time = []
        for i in range(len(b[0])):
            plus = 0
            for p in range(len(arr)):
                plus+=arr[p]*b[p][i]
            time.append(plus)
        main.append(time)
    return main

make_weights = lambda shape: [[0 for j in range(shape[1])] for i in range(shape[0])]

sigmoid = lambda x: [1 / (1 + E**(-k[0])) for k in x]

sigmoid_derivative = lambda x: [k * (1 - k) for k in x]

T = lambda a: [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

def main():
    weights = make_weights((3, 1))

    train_x = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
    train_y = [0, 1, 1, 0]

    for _ in range(1500):
        output = sigmoid(dot(train_x, weights))

        error = [y-p for y, p in zip(train_y, output)]

        delta = [[e*d] for e, d in zip(error, sigmoid_derivative(output))]

        adjustment = dot(T(train_x), delta)

        for i in range(len(weights)):
            weights[i][0] += adjustment[i][0]

    print(sigmoid(dot([[1, 0, 1]], weights)))

if __name__ == '__main__':
    main()