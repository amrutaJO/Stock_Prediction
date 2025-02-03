import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#st.write("hi")

#title 
app_name ='Trading Bot'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company')
#add an image online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

#take input from the user of app about the start and end date

#sidebar
st.sidebar.header('Select the parameters from below')

start_date= st.sidebar.date_input('Start date',date(2020,1,1))
end_date= st.sidebar.date_input('End date',date(2020,12,31))

#ad ticker symbol list
ticker_list =["AAPL","MSFT","GOOG","GOOGL","META","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"] 
ticker = st.sidebar.selectbox('Select the company',ticker_list)


#fetch data from user inputs

data = yf.download(ticker,start=start_date,end=end_date)

#add date as a column to dataFrame
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from',start_date,'to',end_date)
st.write(data)

#plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**NOTE:** Select your specific date range on the sidebar,or zoom i on the plot and select your specific column")

fig = px.line(data, x='Date',y=data.columns,title='Closing price of th stock',width=1000,height=600)
st.plotly_chart(fig)

#add a selct box to slect column from data
column = st.selectbox('Select the column to b used forcasting', data.columns[1:])

#subsetting the data
data = data[['Date',column]]
st.write("Selected data")
st.write(data)

#ADF test check stationarity

st.header('Is data Stationary?')
st.write(adfuller(data[column])[1]<0.05)

#decomposition of data
st.header('Decomposition of the data')
decomposition= seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#make same plot in plotly
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend,title='Trend',width=900,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal,title='Seasonality',width=900,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid,title='Residuals',width=900,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Red',line_dash='dot'))

#lets run the model
#user unput for three parameters for the model and seasonal order
p = st.slider ('select the value of p' , 0,5,2)
d = st.slider ('select the value of d' , 0,5,1)
q = st.slider ('select the value of q' , 0,5,2)
seasonal_order=st.number_input('Select the value of seasonal p' ,0,24,12)

model =sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model= model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

#pridict the future value with user input values
st.write("<p style= 'color:green; font-size:50px; font-weight: bold;'> forecasting the data </p>" ,unsafe_allow_html=True)

forecast_period = st.number_input('Select the number of days to forecast',1,65,10)
#forecast_period = st.number_input("## Enter forecast period in days", value=10)

#predict all values for the forecast period and the current dataset 
predictions= model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions= predictions.predicted_mean
#st.write(len(predictions))
#add index to results dataframes as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions),freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predicates" , predictions)
st.write("## Actual Data", data)

#lets plot the data
fig = go.Figure()
#add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
#add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode='lines',name='Predicted', line=dict(color='red')))
#set the title and axis labels
fig.update_layout(title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Price',width=1000,height=400)
#display the plot
st.plotly_chart(fig)

#Add buttons to show and hide seperate plots
show_plots = False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title='Actual',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title='Predicted',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
        show_plots = True
    else:
        show_plots = False
#add hide plots button
hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plots = False

st.write("---")


