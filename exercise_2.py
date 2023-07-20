import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analysis(name, data):
  """
  Plot Zipfian distribution of words + true Zipfian distribution. Compute MSE.

  :param name: title of the graph
  :param data: list of words
  """
  # create dataframe with tokens
  data = pd.DataFrame(data, columns = ['words'])

  # compute frequency table and sort in descending order
  freq_table = pd.crosstab(index=data['words'], columns='count')
  freq_table = freq_table.sort_values(by=['count'], ascending = False)
  
  # convert frequency table to numpy array and compute rank and zipf list
  freq_list = freq_table.values.flatten()
  rank_list = np.arange(1, len(freq_list)+1, 1, dtype = int)
  zipf_list = freq_list[0] / rank_list

  # compute MSE
  mse = np.mean(np.square(freq_list - zipf_list))

  # take logarithm of zipf and frequency lists
  zipf_list = np.log(zipf_list)
  freq_list = np.log(freq_list)
  rank_list = np.log(rank_list)

  # plot graph
  plt.figure(figsize = (5,5))
  plt.scatter(rank_list, freq_list, label = f'{name}_curve')
  plt.scatter(rank_list, zipf_list, label = f'zipf_curve')
  plt.xlabel('Rank (log-scale)')
  plt.ylabel('Frequency (log-scale)')
  plt.title (f'Frequency vs Rank for {name}')
  plt.legend()
  plt.show()
  
   # print MSE
  print(f'MSE for {name} text : {mse:.10f}')
