# -*- coding: utf-8 -*-
"""GAF-Energy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZVdIRgOK950sceIfgdFRcSPK9t3A57Kv
"""

# Import data
file_id = '172-tCAy4AEFKEaRD3tLVPZCimmow6L4p' #'1THG3uj2h523bk8CvgEA9vw-SGDknXePp' 
filename = 'dataset_odroid_ai_lab.csv' # 'dataset_odroid_energy_lab.csv'
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
dataset_downloaded = drive.CreateFile({'id': file_id})
dataset_downloaded.GetContentFile(filename)  
import pandas as pd
col_name = ['date','temp','hum','lux','bar']
df = pd.read_csv(filename, names=col_name, header=None)
df.drop(index=df.index[0], 
        axis=0, 
        inplace=True)
convert_dict = {'date': str,
                'temp': float,
                'hum': float,
                'lux': float,
                'bar': float
               }
df = df.astype(convert_dict)
# print(df.head())

!pip install pyts

import os
print('Creating Directories:')
PATH = os.path.abspath('')
GAF = os.path.join(PATH , 'GramianAngularFields')
TRAIN_PATH = os.path.join(GAF , 'TRAIN_ENERGY')
TRAIN_LONG = os.path.join(TRAIN_PATH , 'LONG')
TRAIN_SHORT = os.path.join(TRAIN_PATH , 'SHORT')
os.makedirs(TRAIN_LONG, exist_ok=True)
os.makedirs(TRAIN_SHORT, exist_ok=True)
DATA_PATH = os.path.join(PATH, 'TimeSeries')
os.makedirs(DATA_PATH, exist_ok=True)
MODELS_PATH = os.path.join(PATH, 'Models')
os.makedirs(MODELS_PATH, exist_ok=True)
print(GAF)
print(TRAIN_PATH)
print(TRAIN_LONG)
print(TRAIN_SHORT)
print(DATA_PATH)
print(MODELS_PATH)

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
import pandas as pd
import os
from typing import *

matplotlib.use('Agg')


# Pass times-eries and create a Gramian Angular Field image
# Grab times-eries and draw the charts
def create_gaf(ts) -> Dict[str, Any]:
    """
    :param ts:
    :return:
    """
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    return data


# Create images of the bundle that we pass
def create_images(X_plots: Any, image_name: str, destination: str, image_matrix: tuple =(2, 2)) -> None:
    """
    :param X_plots:
    :param image_name:
    :param destination:
    :param image_matrix:
    :return:
    """
    fig = plt.figure(figsize=[img * 4 for img in image_matrix])
    grid = ImageGrid(fig,
                     111,
                     axes_pad=0,
                     nrows_ncols=image_matrix,
                     share_all=True,
                     )
    images = X_plots
    for image, ax in zip(images, grid):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='rainbow', origin='lower')

    repo = os.path.join('GramianAngularFields/TRAIN_ENERGY', destination)
    fig.savefig(os.path.join(repo, image_name))
    plt.close(fig)

# import timeseries_to_gaf as ttg
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from multiprocessing import Pool
import pandas as pd
import os
import datetime as dt
from typing import *


PATH = os.path.abspath('')
IMAGES_PATH = os.path.join(PATH, 'GramianAngularFields/TRAIN_ENERGY')
TEST_PATH = os.path.join(PATH, 'GramianAngularFields/TEST_ENERGY')
DATA_PATH = os.path.join(PATH, 'TimeSeries')


def data_to_image_preprocess() -> None:
    """
    :return: None
    """
    print('PROCESSING DATA')
    # Drop unnecessary data_slice
    global df
    df['DateTime'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df = df.groupby(pd.Grouper(key='DateTime', freq='1h')).mean().interpolate().reset_index()     # '1min'
    # print(df)
    # Send to slicing
    set_gaf_data(df)


def set_gaf_data(df: pd.DataFrame) -> None:
    """
    :param df: DataFrame data_slice
    :return: None
    """
    dates = df['DateTime'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = 0
    box_size = 20
    # Container to store data_slice for the creation of GAF
    decision_map = {key: [] for key in ['LONG', 'SHORT']}
    while True:
        if index >= len(list_dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['DateTime'] < list_dates[len(list_dates) - 1]) & (df['DateTime'] > list_dates[index])]
        print("==========================================")
        print(data_slice)
        print("==========================================")
        gafs = []
        # Group data_slice by time frequency
        for freq in ['4h']:
            group_dt = data_slice.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['temp'].tail(box_size))
            gafs.append(group_dt['hum'].tail(box_size))
            gafs.append(group_dt['bar'].tail(box_size))
            gafs.append(group_dt['lux'].tail(box_size))
        # Decide what trading position we should take on that day
        # future_value = df[df['DateTime'].dt.date.astype(str) == list_dates[index]]['temp'].iloc[-1]
        # current_value = data_slice['temp'].iloc[-1]
        # decision = trading_action(future_close=future_value, current_close=current_value)
        decision_map['SHORT'].append([list_dates[index - 1], gafs])
        index += 1
    
    print('GENERATING IMAGES')
    # Generate the images from processed data_slice
    generate_gaf(decision_map)
    # Log stuff
    dt_points = dates.shape[0]
    total_short = len(decision_map['SHORT'])
    total_long = len(decision_map['LONG'])
    images_created = total_short + total_long
    print("========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}".format(dt_points,
                                                                           images_created))


def generate_gaf(images_data: Dict[str, pd.DataFrame]) -> None:
    """
    :param images_data:
    :return:
    """
    for decision, data in images_data.items():
        for image_data in data:
            to_plot = [create_gaf(x)['gadf'] for x in image_data[1]]
            create_images(X_plots=to_plot,
                              image_name='{0}'.format(image_data[0].replace('-', '_')),
                              destination=decision)


if __name__ == "__main__":
    pool = Pool(os.cpu_count())
    print(dt.datetime.now())
    print('CONVERTING TIME-SERIES TO IMAGES')
    pool.apply(data_to_image_preprocess)
    print('DONE!')
    print(dt.datetime.now())