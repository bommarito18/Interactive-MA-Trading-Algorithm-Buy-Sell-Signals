#!/usr/bin/env python
# coding: utf-8

# In[26]:


#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go

plt.style.use('seaborn-pastel')


# In[27]:


stock = input("Enter Ticker: ")


# In[28]:


MovAv1 = int(input("MA: "))


# In[29]:


MovAv2 = int(input("MA: "))


# In[30]:


Start = input(" Year-Month-Day: ")


# In[31]:


df = web.DataReader(stock, data_source='yahoo', start=Start)
df


# In[32]:


df['MA'] = df['Close'].rolling(MovAv1).mean()
df['MA1'] = df['Close'].rolling(MovAv2).mean()

MA = df['MA']
MA1 = df['MA1']


# In[33]:


df['Date'] = pd.date_range(start=Start, periods=len(df), freq='B')


# In[34]:


fig = go.Figure([go.Scatter(x=df['Date'], y=df['Close'])])
fig.show()


# In[35]:


from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['MA'], name="MA1"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['MA1'], name="MA2"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig.update_xaxes(title_text="xaxis title")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig.show()


# In[36]:


#Create a function to signal when to buy and sell an asset
def buy_sell(signal):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(0,len(signal)):
    #if MA > MA1  then buy else sell
      if signal['MA'][i] > signal['MA1'][i]:
        if flag != 1:
          sigPriceBuy.append(signal['Close'][i])
          sigPriceSell.append(np.nan)
          flag = 1
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('Buy')
      elif signal['MA'][i] < signal['MA1'][i]:
        if flag != 0:
          sigPriceSell.append(signal['Close'][i])
          sigPriceBuy.append(np.nan)
          flag = 0
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('sell')
      else: #Handling nan values
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
  
  return (sigPriceBuy, sigPriceSell)


# In[37]:


#Create a new dataframe
signal = pd.DataFrame(index=df['Close'].index)
signal['Close'] = df['Close']
signal['MA'] = MA
signal['MA1'] = MA1


# In[38]:


signal


# In[39]:


x = buy_sell(signal)
signal['Buy_Signal_Price'] = x[0]
signal['Sell_Signal_Price'] = x[1]


# In[40]:


signal


# In[41]:


#Daily returns Data
stock_daily_returns = df['Adj Close'].diff()


# In[42]:


df['cum'] = stock_daily_returns.cumsum()
df.tail()


# In[43]:


# MA > MA1 Calculation
df['Shares'] = [1 if df.loc[ei, 'MA']>df.loc[ei, 'MA1'] else 0 for ei in df.index]


# In[44]:


df['Close1'] = df['Close'].shift(-1)
df['Profit'] = [df.loc[ei, 'Close1'] - df.loc[ei, 'Close'] if df.loc[ei, 'Shares']==1 else 0 for ei in df.index]


# In[45]:


#Profit per Day, and Accumulative Wealth
df['Wealth'] = df['Profit'].cumsum()
df.tail()


# In[46]:


df['diff'] = df['Wealth'] - df['cum']
df.tail()


# In[47]:


df['pctdiff'] = (df['diff'] / df['cum'])*100
df.tail()


# In[48]:


start = df.iloc[0]
df['start'] = start
start['Close']


# In[49]:


start1 = start['Close']
df['start1'] = start1
df


# In[52]:


my_stocks = signal
ticker = 'Close'

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(x=df['Date'], y=df['cum'], name="Buy Hold Profit",
    marker=dict(color="Pink"),
    line = dict(color = "#ff425b"),
    stackgroup = "one"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y= df['Wealth'], name='MA Profit',
    marker=dict(color="Yellow"),
    line = dict(color = "#7eb8fc"),
    stackgroup = "two",
    opacity = 0.6),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y= my_stocks['Buy_Signal_Price'], name='Buy Signal',
    marker=dict(color="Green", size=12),
    mode="markers",) ,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y= my_stocks['Sell_Signal_Price'], name='Sell Signal',
    marker=dict(color="Red", size=12),
    mode="markers",) ,
    secondary_y=False,
)


fig.add_trace(
    go.Scatter(x=df['Date'], y=df['MA1'], name="MA2"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['MA'], name="MA1"),
    secondary_y=False,
)


fig.add_trace(
    go.Ohlc(
    x=df['Date'],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='OHLC',
    increasing_line_color= 'cyan', decreasing_line_color= 'gray'),
    secondary_y=False
)

# Add figure title
fig.update_layout(
   #title_text="Interactive Buy / Sell"
   title_text=('1 Share of Stock from MA(1) > MA (2) Strategy is : ${:.2f}'.format(df.loc[df.index[-2],'Wealth']))
)

# Add figure title
fig.update_layout(
   #title_text="Interactive Buy / Sell"
   title_text=('MA (1) > MA (2) Difference Compared to Buy & Hold is : ${:.2f}'.format(df.loc[df.index[-2],'diff']))
 
)

# Set x-axis title
fig.update_xaxes(title_text="<b>Year</b>")


# Set y-axes titles
fig.update_yaxes(title_text="<b>USD($)</b>", secondary_y=False)
#fig.update_yaxes(title_text="Year", secondary_y=True)

fig.update_layout(legend_title_text= 'Stock Close Starts at : ${:.2f}'.format(df.loc[df.index[-2],'start1']))

fig.update_xaxes(rangeslider_visible=True)


fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.update_xaxes(rangeslider_visible=True)

fig.update_layout( width=1500, height=700)

#fig.update_layout(
#    annotations=[dict(
#        x='2020-03-25', y=0.05, xref='x', yref='paper',
#        showarrow=False, xanchor='left', text='Increase Period Begins')]
#)

#fig.update_layout(plot_bgcolor='rgb(203,213,232)')

fig.show()


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])


app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


# In[ ]:




