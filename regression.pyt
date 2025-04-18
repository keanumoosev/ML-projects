import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])
m=0
c=0
learning_rate=0.01
epochs=1000
n=float(len(x))
for i in range(int(epochs)):
    y_pred=m*x+c
    loss=(1/n)*sum((y-y_pred)**2)
    D_m=(-2/n)*sum(x*(y-y_pred))
    D_c=(-2/n)*sum(y-y_pred)
    m=m-learning_rate*D_m
    c=c-learning_rate*D_c
    if i%100==0:
        print(f'Epoch{i}: Loss={loss}')
print(f"Final slope (m): {m}")
print(f"Final intercept (c): {c}")
plt.scatter(x,y,color='blue')
plt.plot(x,m*x+c,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression from Scratch')
plt.show()
