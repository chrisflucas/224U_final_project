###################################
# Author: Ramin Ahmari (Jun 2018) #
###################################

import pandas as pd
import numpy as np 
import csv

df = pd.read_csv('category_parser.csv')
	
rac_mean = df["Rude/Agressive_C"].mean()
emc_mean = df["Emotive_C"].mean()
pec_mean = df["Perceptive_C"].mean()
soc_mean = df["Social_C"].mean()
frc_mean = df["Friendly_C"].mean()
nec_mean = df["Negative_C"].mean()
poc_mean = df["Positive_C"].mean()
coc_mean = df["Confrontational_C"].mean()
inc_mean = df["Indecisive_C"].mean()
sec_mean = df["Selfish_C"].mean()
rai_mean = df["Rude/Agressive_I"].mean()
emi_mean = df["Emotive_I"].mean()
pei_mean = df["Perceptive_I"].mean()
soi_mean = df["Social_I"].mean()
fri_mean = df["Friendly_I"].mean()
nei_mean = df["Negative_I"].mean()
poi_mean = df["Positive_I"].mean()
coi_mean = df["Confrontational_I"].mean()
ini_mean = df["Indecisive_I"].mean()
sei_mean = df["Selfish_I"].mean()

mean_list = [rac_mean, emc_mean, pec_mean, soc_mean, frc_mean, nec_mean, poc_mean, coc_mean, inc_mean, sec_mean, rai_mean, emi_mean, pei_mean, soi_mean, fri_mean, nei_mean, poi_mean, coi_mean, ini_mean, sei_mean]

f = open('category_parser_averaged.csv', 'w')
with f:
	writer = csv.writer(f)
	average_row = []
	for indx, row in df.iterrows():
		for idx, elem in enumerate(row):
		  if elem > mean_list[idx]:
		  	average_row.append(1)
		  else:
		  	average_row.append(0)
		writer.writerow(average_row)
		average_row = []