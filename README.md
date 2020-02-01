# DayTrader

## Quickstart
https://colab.research.google.com/drive/1JKqaOCajPgmLiPmP-JzJSS8VBnCSwNF6#offline=true&sandboxMode=true


## Context
Predicting the stock market is a game as old as the stock market itself.  On popular ML platforms like Kaggle, users often compete to come up with highly nuanced, optimized models to solve the stock market starting just from price data.  LSTMs may be the solution, but the real problem isn't the models - it's the data.  

Human and algorithmic traders in the financial industry know this, and augment their datasets with lots of useful information about stocks called "technical indicators".  These indicators have fancy sounding names - e.g. the "Aroon Oscillator" and the "Chaikin Money Flow Index", but most boil down to simple calculations involving moving averages and volatility.  Access to these indicators is unrestricted for humans (you can view them on most trading platforms), but access to well formatted indicators (not just visual lines on a screen) for large datasets reaching back significantly in time is nearly impossible to find.  Even if you pay for a service, API usage limits make putting together such a dataset prohibitively expensive.

The fact that this information is largely kept behind paywalls for large firms with proprietary resources makes me question the fairness of this market.  With a data imbalance like this, how can a single trader - a daytrader - expect to make money?  I wanted to make this data available to the ML community because it is my hope that bringing this data to the community will help to even the scales.  Whether you're just looking to toy around and make a few bucks, or interested in contributing to something larger - a group of people working to develop algorithms to help the "little guy" trade - I hope this dataset will be helpful. To the best of my knowledge, this is the first dataset of its kind, but I hope it is not the last.

## Data
I'd recommend starting here:\
https://colab.research.google.com/drive/1JKqaOCajPgmLiPmP-JzJSS8VBnCSwNF6#offline=true&sandboxMode=true

In the above notebook, I've uploaded the data and made it available for read access.  The notebook will walk you through processing the data and putting it to work building advanced ML models.

If you'd like to download it directly, you can do so here (31GB):
https://drive.google.com/file/d/1HVKWtLWlIZj5yY4T3R1CDSn3c_Nh1lB5/view?usp=sharing

## Acknowledgements
* The many online tutorials and specifications which helped me write and test the indicator functions
* [borismarjanovic](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) for making public an amazing dataset that I use as a baseline for the colab notebook and the direct download file above
* The many online services that have allowed me to download all the recent price information to augment Boris' dataset.

## Next Steps / Future Directions
* Building inventive models using this dataset to more and more accurately predict stock price movements
* Incorporating arbitrage analysis across stocks
* Hedging
* Options and selling short

## Collaboration
If this interests you, reach out!  My email is abwilf [at] umich [dot] edu.  The repository I used to generate the dataset is here:

