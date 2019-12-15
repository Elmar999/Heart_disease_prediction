import numpy as np
import Data_manipulation as Dp
import pandas as pd


path = "disease.csv"


data = pd.read_csv(path , sep=";")

matrix = data.to_numpy()