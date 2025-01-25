import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from utils import get_season
from datetime import datetime


czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]
frankfurt_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

def enrich_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)

    # Number of days until next holiday
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    # Fill NaT values by using the next valid observation to fill the gap
    df['next_holiday_date'] = df['next_holiday_date'].bfill() 
    df['date_days_to_next_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)

    # Number of days until shops are closed
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['date_days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)

    # Was the shop closed yesterday?
    df['day_after_closed_day'] = ((df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Are shops closed today and were they also closed yesterday (e.g., December 26 in Germany)?
    df['second_closed_day'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Was the shop closed the last two days?
    df['day_after_two_closed_days'] = ((df['shops_closed'] == 0) & (df['second_closed_day'].shift(1) == 1)).astype(int)
    

    return df

calendar = pd.read_csv('csv_input/calendar.csv', parse_dates=['date'])

calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frankfurt_holidays)

calendar_enriched = pd.DataFrame()

for location in ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']:
    calendar_enriched = pd.concat([
        calendar_enriched,enrich_calendar(calendar.query('date >= "2020-08-01 00:00:00" and warehouse ==@location'))])

calendar_enriched.to_csv('csv_junk/calendar_enriched.csv', index=False)
