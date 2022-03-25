import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import datetime as dt
from lifetimes.plotting import plot_period_transactions
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# Reading Data from Database
############################

# credentials.
creds = {'user': '****',
         'passwd': '*****',
         'host': '*****',
         'port': '*****',
         'db': '*****'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
df = retail_mysql_df.copy()

# Select UK customers.
df = df[df["Country"].str.contains("United Kingdom")]
df["Country"].head()

############################
# Functions
############################
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

############################
# EDA & PREP
############################

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df.columns

# Suppressing outliers
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

############################
# CLTV Metrics
############################
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.head()
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[cltv_df["monetary"] > 0]

### Weekly values for recency and T ###
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["frequency"].astype(int)
cltv_df.info()

##################################
# BG-NBD Model
##################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# Top 10 customers we expect to purchase the most in a week
cltv_df['expected_purc_1_week'] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)
cltv_df.sort_values("expected_purc_1_week", ascending=False).head(20)

# Top 10 customers we expect to purchase the most in a month
cltv_df['expected_purc_1_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)
cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)

#################################
# Expected Number of Sales of the Whole Company in 3 Months
#################################
bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

########################################
# Evaluation of prediction results
########################################

plot_period_transactions(bgf)
plt.show()

########################################
# GAMMA GAMMA SUBMODEL (EXPECTED PROFIT)
########################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

#Top 10 most valuable customers
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                         cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(20)

##############################################################
# Calculation of 6-month CLTV by BG-NBD and GG model.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # for 6 months
                                   freq="W",  # Frequency of T.
                                   discount_rate=0.01)
cltv.head()
cltv=cltv.reset_index()
cltv.sort_values(by='clv', ascending=False, inplace=False).head()
cltv_df_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_df_final.head()
cltv_df_final.sort_values(by="clv", ascending=False).head(20)

##############################################################
# 1-month and 12-month CLTV values for 2010-2011 UK customers.
##############################################################
# For 1 months
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # for 1 months
                                   freq="W",  # Frequency of T.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(12)
cltv_df_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_df_final.head()

# For 12 months
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # for 12 months
                                   freq="W",  # Frequency of T.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(12)
cltv_df_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_df_final.head()

##############################################################
# Segment all customers into 3 segments based on 6-month CLTV for 2010-2011 UK customers.
##############################################################
# For 6 months

cltv_6_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # for 6 months
                                   freq="W",  # Frequency of T.
                                   discount_rate=0.01)

cltv_6_month=cltv_6_month.reset_index()
cltv_6_month_final = cltv_df.merge(cltv_6_month, on="Customer ID", how="left")
cltv_6_month_final.sort_values(by="clv", ascending=False).head()
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_6_month_final[["clv"]])
cltv_6_month_final["scaled_clv"] = scaler.transform(cltv_6_month_final[["clv"]])
cltv_6_month_final["segment"] = pd.qcut(cltv_6_month_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

#Top 20 percent for CLTV
cltv_6_month_final.shape[0] *0.20
# 514.0
cltv_6_month_final["top_flag"] = 0
cltv_6_month_final["top_flag"].iloc[0:515] = 1
cltv_6_month_final.head()

# CLTV of segment A (%)
cltv_6_month_summary = cltv_6_month_final.groupby("segment").agg({"count", "mean", "sum"})
cltv_6_month_summary.head()
cltv_6_month_summary.loc["A","clv"]["sum"] / (cltv_6_month_summary.loc["A","clv"]["sum"]+
                                              cltv_6_month_summary.loc["B","clv"]["sum"]+
                                              cltv_6_month_summary.loc["C","clv"]["sum"]+
                                              cltv_6_month_summary.loc["D","clv"]["sum"])

################################
# Send final table to database
################################
cltv_6_month_final.to_sql(name='bugra_varol', con=conn, if_exists='replace', index=False)
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from bugra_varol limit 10", conn)

