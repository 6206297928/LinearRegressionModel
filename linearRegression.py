class Linear_Regression():
#initiating the parameters
  def __init__(self,learning_rate, no_of_iterations):#hyperparameters


    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations


  def fit(self,X,Y):# X will be no of experience and Y will be the no of salary

    #number of training example(count(30)) & number of features(columns(1)) here x is only 1 feature and Y is the target variable

    self.m,self.n=X.shape #the number of row(m)(30) and coulmns(n)(1)

    #initiating the weight and bias
    self.w = np.zeros(self.n)
    self.b=0
    self.X =X
    self.Y=Y

    #implementing Gradient Descent
    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    Y_prediction= self.predict(self.X)

    #calculate the gradients
    dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
    db = - 2 * np.sum(self.Y - Y_prediction)/self.m

    #updating the weights

    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db

  def predict(self,X):
     return X.dot(self.w)+ self.b # Xw + b
  

from sklearn.linear_model import LogisticRegression
model= LogisticRegression()

#importing the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#loading a data from csv to a pandas dataframe
salary_data=pd.read_csv('/content/Salary_dataset.csv')
salary_data.head()

#checking for missing values
salary_data.isnull().sum()

#delete column unnamed
salary_data=salary_data.drop(['Unnamed: 0'],axis=1)

salary_data.isnull().sum()
salary_data.shape

X=salary_data.iloc[:,:-1].values
Y=salary_data.iloc[:,1].values
print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
model=Linear_Regression(learning_rate=0.02,no_of_iterations=1000)#here we can import our pre build lin reg model class by lin_Reg_sample_code.LInear_Regression(and add parameter) dont forget to import that lin_Reg_sample_code.py file
model.fit(X_train,Y_train)

#printing the parameter values(weight and bias)
print('weight = ',model.w[0])
print('bias = ',model.b)


test_data_prediction=model.predict(X_test)

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test,test_data_prediction, color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()