import pandas as pd
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = pd.read_csv('./data/salary_data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1:].values

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=0)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_hat = model.predict(X_test)
    model.summary(Y_test, y_hat)

    # Plotting Linear Regression
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, y_hat, color='red')
    plt.title('Linear Regression')
    plt.xlabel('Salary')
    plt.ylabel('YearsExperience')
    plt.show()
