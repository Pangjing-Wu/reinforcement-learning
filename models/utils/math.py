argmax = lambda arr, eps: [i for i, val in enumerate(arr) if val == max(arr)][0]

float_eq = lambda a, b, eps: abs(a - b) < eps
float_argmax = lambda arr, eps: [i for i, val in enumerate(arr) if abs(val - max(arr)) < eps][0]