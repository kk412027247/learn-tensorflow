from __future__ import print_function
import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
    print('TF import with eager execution!')
except ValueError:
    print('TF already imported with eager execution!')

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

print('primes: ', primes)

ones = tf.ones([6], dtype=tf.int32)
print('ones: ', ones)

just_beyond_primes = tf.add(primes, ones)
print('just_beyond_primes: ', just_beyond_primes)

twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print('primes_doubled: ', primes_doubled)

some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print(some_matrix)
print('\nvalue of some_matrix is: \n', some_matrix.numpy())

scalar = tf.zeros([])

vector = tf.zeros([3])

matrix = tf.zeros([2, 3])

print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.numpy())
print('vector has shape', vector.get_shape(), 'and value: \n', vector.numpy())
print('matrix has shape', matrix.get_shape(), 'and value: \n', matrix.numpy())

primes = tf.constant([2, 3, 4, 5, 11, 13], dtype=tf.int32)
print('primes: ', primes)

one = tf.constant(1, dtype=tf.int32)
print('one: ', one)

just_beyond_primes = tf.add(primes, one)
print('just_beyond_primes: ', just_beyond_primes)

two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * two
print('primes_doubled: ', primes_doubled)

neg_one = tf.constant(-1, dtype=tf.int32)
just_under_primes_squared = tf.add(tf.pow(primes, 2), neg_one)
print('just_under_primes_squared', just_under_primes_squared)

x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)
y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

matrix_multiply_result = tf.matmul(x, y)
print(matrix_multiply_result)

matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)

reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])
print('Original matrix (8x2): ', matrix.numpy())
print('Reshaped matrix (2x8):', reshaped_2x8_matrix.numpy())
print('Reshaped matrix(4x4):', reshaped_4x4_matrix.numpy())

reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])
one_dimensional_vector = tf.reshape(matrix, [16])
print('Reshaped matrix(2x2x4): ', reshaped_2x2x4_tensor.numpy())
print('1-D vector: ', one_dimensional_vector.numpy())

a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])

reshaped_a = tf.reshape(a, [2, 3])
reshaped_b = tf.reshape(b, [3, 1])

result = tf.matmul(reshaped_a, reshaped_b)
print(result)

v = tf.contrib.eager.Variable([3])
w = tf.contrib.eager.Variable(tf.random.normal([1, 4], mean=1.0, stddev=0.35))

print('V: ', v.numpy())
print('w: ', w.numpy())

tf.assign(v, [7])
print(v.numpy())

v.assign([5])
print(v.numpy())

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print(v.numpy())

try:
    print('Assigning [7,8,9] to v')
    v.assign([7, 8, 9])
except ValueError as e:
    print('Exception: ', e)

die1 = tf.contrib.eager.Variable(
    tf.random.uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
)

die2 = tf.contrib.eager.Variable(
    tf.random.uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
)

dice_sum = tf.add(die1, die2)
resulting_matrix = tf.concat(values=[die1, die2, dice_sum], axis=1)

print(resulting_matrix)
