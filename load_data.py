import numpy as np

# 100k
matrix = np.zeros((944, 1683))
with open('ml-100k/u.data', 'r') as fp:
    for line in fp:
        lis = list(map(int, line.split('\t')))
        matrix[lis[0]][lis[1]] = lis[2]

np.save('rating-100k.npy', matrix)


# 100k u1
matrix = np.zeros((944, 1683))
with open('ml-100k/u1.base', 'r') as fp:
    for line in fp:
        lis = list(map(int, line.split('\t')))
        matrix[lis[0]][lis[1]] = lis[2]

np.save('rating-100k-u1.npy', matrix)


# 1m
matrix = np.zeros((6041, 3953))
with open('ml-1m/ratings.dat', 'r') as fp:
    for line in fp:
        lis = list(map(int, line.split('::')))
        matrix[lis[0]][lis[1]] = lis[2]

np.save('rating-1m.npy', matrix)
print(matrix[6040][1097])

