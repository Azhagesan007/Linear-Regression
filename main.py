import pandas
from pandas import DataFrame
import matplotlib.pyplot as pil
from sklearn.linear_model import LinearRegression

data = pandas.read_csv("cost_revenue_clean.csv")


# print(data.describe())

X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])
# print(X)
# print(Y)
pil.figure(figsize=(10, 6))
pil.xlim(0, 450000000)
pil.ylim(0, 3000000000)
pil.title('Production Budeget Vs Global Revenue')
pil.xlabel("Production Budeget")
pil.ylabel("Global Revenue")
pil.scatter(X, Y, alpha=0.3)


regression = LinearRegression()
regression.fit(X, Y)
pil.plot(X, regression.predict(X), color="red", linewidth= 4)
pil.show()


single_production_budget = int(input("Enter the production : \n"))

predicted_global_revenue = regression.predict([[single_production_budget]])
print("Production Budget: {}, Predicted Global Revenue: {}".format(single_production_budget, predicted_global_revenue[0][0]))