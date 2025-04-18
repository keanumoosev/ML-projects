import numpy as np
x=np.array([[2,3],
            [1,5],
            [2,8],
            [5,1],
            [6,2],
            [7,3]
            ])
y=np.array([0,0,0,1,1,1])
weights=np.random.rand(2)
bias=np.random.rand(1)
learning_rate=0.01
def step(x):
    return 1 if x>=0 else 0
for epoch in range(100):
    total_error=0
    for i in range(len(x)):
        linear_output=np.dot(x[i],weights)+bias
        prediction=step(linear_output)
        error=y[i]-prediction
        weights+=learning_rate*error*x[i]
        bias+=learning_rate*error
        total_error+=abs(error)
    if total_error==0:
        break
print("weights",weights)
print("bias",bias)
test_point=np.array([4,3])
test_result=step(np.dot(test_point,weights)+bias)
print(f"Test Point {test_point} classified as: {test_result}")
