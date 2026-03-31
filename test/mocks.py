import torch as to

BN1 = to.Tensor(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
).int()
BN2 = to.Tensor(
    [
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
).int()

BN3 = to.Tensor(
    [
        [0 ,0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ]
).int()

NOTABN1 = to.Tensor(
    [
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]
).int()

NOTABN2 = to.Tensor(
    [
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
).int()

MAXAR = 3

AR1 = to.Tensor([0, 1, 0, 2, 3]).int()
AR2 = to.Tensor([1, 1, 1, 1, 1]).int()

BN1_W = [
    to.Tensor([0.1, -0.1]),
    to.Tensor([0.2, -0.1, 0.01]),
    to.Tensor([-0.2,  0.1]),
    to.Tensor([0.1, -0.2, 0.03, -0.04]),
    to.Tensor([0.2, 0.01, 0.03, -0.01]),
]
BN2_W = [
    to.Tensor([0.5, -0.2, 0.3, -0.1, 0.1, 0.01]),
    to.Tensor([0.5, -0.2, 0.3, -0.1, 0.01]),
    to.Tensor([0.5, -0.2, 0.3, 0.01]),
    to.Tensor([0.5, -0.2, 0.01]),
    to.Tensor([0.5, 0.01]),
]
BN3_W = [
    to.Tensor([1.0]),
    to.Tensor([-3.5, 0.5]),
    to.Tensor([1, -1])
]

SIGMA3_2 = to.Tensor([4, 4, 3])

BATCH = 200
DATAMOCK = to.stack(
    [
        to.normal(to.Tensor([1, 2, -1, -2, 0]), to.Tensor([1, 0.5, 0.7, 1.1, 0.9]))
        for _ in range(BATCH)
    ]
)
