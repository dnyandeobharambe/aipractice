'''
If the dimensions of two arrays are dissimilar, element-to-element operations are not possible. However, operations on arrays of non-similar shapes is still possible in NumPy, because of the broadcasting capability. The smaller array is broadcast to the size of the larger array so that they have compatible shapes.

Broadcasting is possible if the following rules are satisfied −

Array with smaller ndim than the other is prepended with '1' in its shape.

Size in each dimension of the output shape is maximum of the input sizes in that dimension.

An input can be used in calculation, if its size in a particular dimension matches the output size or its value is exactly 1.

If an input has a dimension size of 1, the first data entry in that dimension is used for all calculations along that dimension.

A set of arrays is said to be broadcastable if the above rules produce a valid result and one of the following is true −

Arrays have exactly the same shape.

Arrays have the same number of dimensions and the length of each dimension is either a common length or 1.

Array having too few dimensions can have its shape prepended with a dimension of length 1, so that the above stated property is true.

The following program shows an example of broadcasting.

'''



import numpy as np

x = np.array([0, 0, 0])

y = np.array([[1],
     [1],
     [1],
     [1]])

z = np.array([[1, 1],
     [1, 1],
     [1, 1],
     [1, 1]])


print(x.shape)
print(y.shape)
print(z.shape)

print(x+y)
print(y+z)
## following are not valid 
#print(z+x)
#print(np.matmul((x+y), z))