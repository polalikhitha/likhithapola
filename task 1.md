# Task 1- Prediction Using Supervised ML

# By Polalikhitha

# Data Science and Business Analytics Internship

# GRIP: The spark foundation


```python
# Importing all libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)

s_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.7</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours Vs percentage')
plt.xlabel('Hours Studies')
plt.ylabel('percentage Score')
plt.show()
```


    
![png](output_6_0.png)
    


From the graph,we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score

# Preparing the data

The next step is to divide the data into "attributes"(inputs)and "labels"(outputs)


```python

### Independent and Dependent features
X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, -1].values
```

Now that we have our attributes and labels, the next step is to split this data into training and test sets.We'lldo this by using scikit-learn's built-in train_test_split() method:


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                            test_size=0.2, random_state=0)
```

# Training the Algorithm

We have split our data into training and testing sets,and now is finally the time to train our algorithm


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")
```

    Training complete.
    


```python
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

#Plotting for the test data
plt.scatter(X,y)
plt.plot(X,line);
plt.show()
```


    
![png](output_16_0.png)
    


# Making predictions

Now that we have our algorithm,it's time to make some predictions


```python
print(X_test) #Testing data - In Hours
y_pred = regressor.predict(X_test) #Predicting the scores
```

    [[1.5]
     [3.2]
     [7.4]
     [2.5]
     [5.9]]
    


```python
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>16.884145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>33.732261</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>75.357018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>26.794801</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>60.491033</td>
    </tr>
  </tbody>
</table>
</div>




```python
# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
```

    No of Hours = 9.25
    Predicted Score = 93.69173248737538
    


```python

```
