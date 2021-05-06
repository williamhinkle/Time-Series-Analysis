# Time-Series-Forecast for Waterway Transportation data using 2018 historical data to predict short tons to be carried in the next 2 years
Why time series analysis?
We will be able extract meaningful statistics and other characteristics using the one variable, time. We can use it to predict trends in the future as well as understand past behavior. The past defines what will happen in the future. In order to use a time serious the data must be stationary. We also can’t use a cross-sectional analysis for short tons delivered because the transportation data isn’t consistent over time.

We will start by importing and analyzing our data to obtain a baseline of knowledge as well as plan out our methodologies for forecasting. This comprises the work we’ve been doing for the past 2 weeks.

Cell 1 in github is composed of these lines of Python code that are used to read in the necessary packages to run this time series. They are for data analysis, processing, plotting and visualizing, and pandas/statsmodels which are the libraries that will deal with all of our machine learning. 

Cell 2 is used to read in the 2018 shipping dataset. We read ours in as a CSV file and included the completed year and short tons.
 
Cell 3 in is used to show a preview of the data including completed year and ShortTons as well as import the time packages. Below that is cell 3 out which is this representation in a chart.

Cell 4 in is then used to make the first plot of the data. Line one sets the x label to be the date while line two sets the y label to be short tons delivered. Line three actually plots the data on the plot. Cell 4 out is the plot of the data.

Cell 5 is where we first start adding key analysis components to the plot to allow us to have a good training set. This training set will allow us to make predictions into the future. Line one is making a rolling mean of the short tons for each time segment. Line 2 is the same thing but using a rolling standard deviation. Lastly, line 3 simply prints out a section of the data to test the outcome. The window size allows us to set the duration of the rolling mean and standard deviation sets, in this case 12 for the amount of months in the year we’re analyzing. 

Cell 6 is then executed to plot the new rolling standard deviation and mean overlaid onto the original data. The blue line represents the original data. The red line represents the rolling mean and the black line represents the rolling standard deviation.

Cell 7 is then used to test the stationarity of these two rolling functions with the Dickey-Fuller test. The test shows a variety of stats but the ones we were focusing on were the p-value and critical values. The p-value should always be less than the previous iteration and the critical value at 1% should be as close as possible to the test statistic. We want to get the red line, which represents the rolling mean, as flat as possible. The flatness of the line is a proxy for the stationarity of the data and we use the statistics as key indicators. 


We are now going to edit the way in which the data is represented to further straighten the rolling mean line which will in turn increase the stationarity of the data when fed into our Machine Learning algorithm. 

Cell 8 in is executed which alters the y-value of the plotted data to be logarithmic. This log scale allows for the data to be more easily interpreted. This is represented by the plot in Cell 8 out.

Cell 9 in calculates and plots the new rolling mean and standard deviations. Cell 9 out shows us how these have changed compared to the previous iteration.

Cell 10 in is another method to make the line more stationary. Here we take the difference between the rolling mean and rolling standard deviation and any N/A’s are removed. In cell 10 out the head of the set is then output to give us an idea of what changed.

Cell 11 defines a test that we can call on in later cells to observe the change in standard deviation and moving average. This part of the analysis requires a lot of cleaning to find the optimal state of the data to be put into the machine learning, so a set of code we can refer back to will save us time.

Cell 12 is simply plotting the results from the work we did in cell 11 to test the change in stationarity. We can see in the chart at the bottom that the p-value has again decreased which is exactly what we want for a good training set, and we now know that our test method is effective.

Jake
Cell 13 in again alters the data by using an exponential decay. Cell 13 out plots this change.

Cell 14 calculates the log scale minus the weighted average and tested the results using our stationarity test. As you can see again the red line is getting flatter and the p-value is continuing to decrease.

Cell 15 in again alters the data by using the shift function to shift all of the values. Cell 15 out shows the plot of this change without the mean or standard deviation

Cell 16 we drop the N/A’s and plot the data again with the new iteration. We can see that the rolling mean is almost completely flat which means it is stationary and ready for projection. Continued interactions with the data can be made but will not make it any more stationary or improve forecasting.

Cell 24 we import the statsmodel package to help us analyze the prepared data. It also allows us to break the data into 4 components: the original set, the trend data (positive or negative), the seasonality of the data, and any interfering residuals. It’s clear that seasonality is going to have an impact on our forecast. 

Cell 25 we now check the residuals and drop any outlying values to see if it impacts the stationarity of the data or not. This can be thought of as the normalization portion, as seen when comparing the outputs before and after residuals are modified. 

It’s worth noting that we ran into some errors when parsing the ShortTon data, however we believe through our troubleshooting and research that it was likely a compatibility issue between some of the libraries we were using.

Cell 26 creates the autocorrelation and partial autocorrelation graphs labeled ACF and PACF respectively. These graphs validate that our p-value is adequate for our forecasting model. The rate at which the graph initially drops to zero is the indicator for the quality of the p-value. In both graphs, this happens rapidly and we can assume our data is properly prepared for Machine Learning. The ACF and PACF plots are shown in the graphs for cell 26.

This next part is the crux of the project, where we build models for our data to be trained. 

The method we will be using is called ARIMA. This is pulled from the statsmodel package and is the primary reason we chose to program in python. This method allows us to do time series forecasting rather than cross sectional analysis. It is broken down into 3 components: AR which stands for auto regression and uses historical data to predict patterns, MA which stands for moving average, which is the rolling mean value we were trying to straighten and represents the stationarity, and I which is our integration component that ties the two together. It is the differentiation of our prediction curve, which is 1 in our case because time is linear.

Cells 39 and 42 are the AR and MA functions respectively, and playing with the ordered pair (1,1,2) gives us the lowest possible RSS value (error between the functions). 

Cell 44 in is the culmination of all of our work so far, and it takes the AR and MA inputs to create the model we will use for forecasting. Cell 44 out shows the mapping of our forecasted function to our actual dataset for 2018.

Cells 45, 48, 50, 51, and 52 all work to undo the modifications we originally made to the data so that our model can be accurate to the 2018 dataset. Fitted values, cumulative sums, indexing, and exponentiation are all used to achieve this.

Cell 53 is our prediction using our model for the next 2 years (730 days) given a confidence interval of 95%.
