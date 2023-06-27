import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
# Good resources for getting and manipulating dates
# * https://www.programiz.com/python-programming/datetime/current-datetime
# * https://stackoverflow.com/questions/6871016/adding-days-to-a-date-in-python
from datetime import date, timedelta

# Summations of the days of previous months (ex: March has 31 + 28 days from January and February), used in daysSinceStartingYear
monthToDaysDict = {
    1: 0,
    2: 31,
    3: 31 + 28,
    4: 31 + 28 + 31,
    5: 31 + 28 + 31 + 30,
    6: 31 + 28 + 31 + 30 + 31,
    7: 31 + 28 + 31 + 30 + 31 + 30,
    8: 31 + 28 + 31 + 30 + 31 + 30 + 31,
    9: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
    10: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
    11: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,
    12: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30
}

# Returns the number of days since a given startYear up to and including the given date, formatted as MM/DD/YYYY
def daysSinceStartingYear(startYear, dateRaw):
    # These particular string splits/indices were necessary for the HTML structure of the given data, see currRowDateText below
    # The intended format is MM/DD/YYYY
    dateComponents = ((dateRaw.split("\n"))[1]).split("/")
    dateMonth = int(dateComponents[0])
    # Assign the number without the leading 0 (if any) and without the comma
    dateDayRaw = dateComponents[1]
    dateDay = int(dateDayRaw[1] if (dateDayRaw[0] == '0') else dateDayRaw[0:2])
    dateYear = int(dateComponents[2])
    # One day is added for every 4 whole years, + 1 day if the current year is a leap year and we are past February
    leapYearsCorrection = ((dateYear - startYear) // 4) + int(((dateYear % 4) == 0) and (dateMonth > 2))
    return ((dateYear - startYear) * 365) + monthToDaysDict[dateMonth] + dateDay + leapYearsCorrection

# Choose the year of the earliest available stock price (useful for combining multiple datasets)
# Here Jan 1 in 2023 with a startingYear of 2023 would count as 1 day
startingYear = 2023

# Fetching the URL with our data and making a parseable object
marketWatchPayPalPage = requests.get('https://www.marketwatch.com/investing/stock/pypl/download-data?mod=mw_quote_tab')
payPalSoup = BeautifulSoup(marketWatchPayPalPage.text, 'html.parser')
print('MarketWatch page URL: https://www.marketwatch.com/investing/stock/pypl/download-data?mod=mw_quote_tab')
print('MarketWatch page status code: ' + str(marketWatchPayPalPage.status_code))

# There are multiple tables with 'aria-label' as "Historical Quotes data table"
# We are only interested in the first one, hence the first element of what find_all returns
payPalHistoricalPricesTableWrapper2 = (payPalSoup.find_all('table', {'aria-label': "Historical Quotes data table"}))[0]
payPalHistoricalPricesTableWrapper1 = payPalHistoricalPricesTableWrapper2.find('tbody', {'class': "table__body row-hover"})
payPalHistoricalPricesTable = payPalHistoricalPricesTableWrapper1.find_all('tr', {'class': "table__row"})

# Collecting day counts and corresponding prices into arrays
payPalPriceDaysArr = np.empty((0))
payPalPricesArr = np.empty((0))
for currRow in payPalHistoricalPricesTable:
    currRowDateText = (currRow.find('td', {'class': "overflow__cell fixed--column"})).text
    payPalPriceDaysArr = np.append(payPalPriceDaysArr, np.array([daysSinceStartingYear(startingYear, currRowDateText)]), axis=0)
    # Using the "Close" price (index 4), the price of the stock when the trading of the stock ends
    # This gets turned into text, and we turn everything except the leading "$" (hence [1:]) into a float
    selectedPrice = float((((currRow.find_all('td', {'class': "overflow__cell"}))[4]).text)[1:])
    payPalPricesArr = np.append(payPalPricesArr, np.array([selectedPrice]), axis=0)

# Creating linear regression model using the arrays, graphing the data with the model's curve
payPalLinRegModel = LinearRegression()
payPalLinRegModel.fit(payPalPriceDaysArr.reshape(-1, 1), payPalPricesArr)
plt.scatter(payPalPriceDaysArr, payPalPricesArr)
plt.plot(payPalPriceDaysArr, payPalLinRegModel.predict(payPalPriceDaysArr.reshape(-1, 1)), color='red')
plt.xlabel('Days since start of the year ' + str(startingYear) + '*')
plt.ylabel('Closing price of one PayPal stock (USD)')
plt.show()
print("*Jan 1, " + str(startingYear) + " is '1' in this axis")

# Predicting unknown prices
print("")
# The leading newline characters format the dates to be similar to that of the original HTML and readable by daysSinceStartingYear
# Adapted from Programiz and StackOverflow
dateTomorrow = '\n' + (date.today() + timedelta(days=1)).strftime("%m/%d/%Y")
dateIn7Days = '\n' + (date.today() + timedelta(days=7)).strftime("%m/%d/%Y")
dateIn30Days = '\n' + (date.today() + timedelta(days=30)).strftime("%m/%d/%Y")
dateIn365Days = '\n' + (date.today() + timedelta(days=365)).strftime("%m/%d/%Y")
predictionTomorrow = payPalLinRegModel.predict((np.array([daysSinceStartingYear(startingYear, dateTomorrow)])).reshape(-1, 1))
predictionIn7Days = payPalLinRegModel.predict((np.array([daysSinceStartingYear(startingYear, dateIn7Days)])).reshape(-1, 1))
predictionIn30Days = payPalLinRegModel.predict((np.array([daysSinceStartingYear(startingYear, dateIn30Days)])).reshape(-1, 1))
predictionIn365Days = payPalLinRegModel.predict((np.array([daysSinceStartingYear(startingYear, dateIn365Days)])).reshape(-1, 1))
print("Predicted PayPal stock price tomorrow: " + str(predictionTomorrow[0]) + " USD")
print("Predicted PayPal stock price 7 days from today: " + str(predictionIn7Days[0]) + " USD")
print("Predicted PayPal stock price 30 days from today: " + str(predictionIn30Days[0]) + " USD")
print("Predicted PayPal stock price 365 days from today: " + str(predictionIn365Days[0]) + " USD")
