import numpy as np
import math, sys, os
import tushare as ts
import pandas as pd
import datetime as dt
from pandas import read_csv
from collections import *
from functools import *

load_from_file = 0

def load_stock_data_from_file(file):
   global load_from_file

   failed = False
   try:
      # 从整理好的文件中获取数据
      load_from_file = 0
      df = read_csv(file, engine='python')
      if len(df.columns) != 5:
         failed = True
   except:
      failed = True

   if failed:
      load_from_file = 1
      failed = False
      try:
         # 从同花顺导出的修改后用逗号分隔的数据中获取数据
         df = read_csv(file, usecols=[0,2,3,4,5], engine='python', header=None)
         if len(df.columns) != 5:
            failed = True
      except:
         failed = True

   if failed:
      # 从同花顺导出的原始数据中获取数据
      load_from_file = 2
      df = read_csv(file, usecols=[0,1,2,3,4], sep='\s+', engine='python')
      df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x[0:10])

   df.columns = ['date', 'open', 'high', 'low', 'close']
   return df

def load_stock_data_from_network(code, kline_type):
   df = ts.get_k_data(code, '19900101', '', ktype=kline_type)
   if len(df) > 0:
      # 把收盘价放最低价后面，保持和同花顺一致顺序
      cols = list(df)
      cols.insert(4, cols.pop(2))
      df = df.loc[:, cols]
      df.drop(columns=['volume', 'code'], inplace=True)
   return df

def load_stock_data_frame(file_or_code, date_start='', date_end=''):
   if is_file_name(file_or_code):
      df = load_stock_data_from_file(file_or_code)
   else:
      df = load_stock_data_from_network(file_or_code, kline_type)

   if len(df) > 0:
      cols = list(df)[1:5]
      df[cols] = df[cols].astype('float32')

      if date_start != '':
         df = df[df[df.columns[0]] >= date_start]

      if date_end != '':
         df = df[df[df.columns[0]] < date_end]

   return df

def detect_cycle_from_time(times):
   v = times[0]
   idx = 0
   for i, x in enumerate(times):
      if x != v:
         idx = i
         break

   return idx

def ktype_from_cycle(cycle):
   if cycle == 1:
      return 'D'
   if cycle == 4:
      return '60'
   if cycle == 8:
      return '30'
   if cycle == 16:
      return '15'
   if cycle == 48:
      return '5'
   else:
      return ''

def find_file(s):
   r = []
   for root, dirs, files in os.walk('.'):
      if root != '.':
         break
      for file in files:
         fn = file.upper()
         if fn.endswith('.TXT'):
            if s[0] == '^':
               if fn.startswith(s[1:]):
                  r.append(file)
            elif s[0] == '~':
               if fn.find('-' + s[1:]) >= 0:
                  r.append(file)
            else:
               if fn.find(s) >= 0:
                  r.append(file)
   return r

def find_file_list(vs):
   r = []
   for v in vs:
      r.extend(find_file(v))
   return r

def is_file_name(f):
   return f.upper().endswith('.TXT')

def code_from_file(file):
   INDEX = ["sh000001", "sh000016"]
   v = [x for x in file.split('-') if (x.isdigit() and len(x) >= 5 and len(x) <= 6) or x in INDEX]
   if len(v) > 0:
      return v[0]
   else:
      return ""

def process_file_update(files):
   updated_file_count = 0
   update_to_date_file_count = 0
   failed_file_list = []

   for file in files:
      code = code_from_file(file)
      if code == '':
         print('Cannot find stock code from file name ', file)
         failed_file_list.append(file)
         continue

      df = load_stock_data_from_file(file)
      ds_date = df.values[:, 0]
      cycle = detect_cycle_from_time(ds_date)
      ktype = ktype_from_cycle(cycle)
      if ktype == '':
         print('Detect cycle ', cycle, ' is not valid for file ', file)
         failed_file_list.append(file)
         continue

      try_count = 0
      while try_count <= 3:
         try_count += 1
         df_net = load_stock_data_from_network(code, ktype)
         if len(df_net) > 0:
            df_net['date'] = df_net['date'].apply(lambda x: x[0:10])
            df_net = df_net[df_net[df_net.columns[0]] > ds_date[-1]]
            break
      else:
         print('Load data from network failed for file ', file)
         failed_file_list.append(file)
         continue

      if len(df_net) > 0:
         df_new = pd.concat([df, df_net])
         df_new.to_csv(file, index=False, float_format='%.3f')
         updated_file_count += 1
         print(file, 'is updated')
      else:
         if load_from_file > 0:
            df.to_csv(file, index=False, float_format='%.3f')
         update_to_date_file_count += 1
         print(file, 'is update to date')

   print("")
   print("Total file: ", len(files))
   print("Updated file: ", updated_file_count)
   print("Update to dated file: ", update_to_date_file_count)
   if len(failed_file_list) > 0:
      print("Failed file: ", len(failed_file_list))
      for f in failed_file_list:
         print("    ", f)

def main():
   argv = sys.argv[1:]

   if len(argv) <= 0:
      print("Usage: stock_data_update file_pattern_list")

   verbose = False
   files_pattern = []

   if True:
      i = 0
      while i < len(argv):
         x = argv[i]

         if x == '-v' or x == '-verbose':
            verbose = True
         elif x[0] != '-':
            files_pattern.append(x.upper())
         else:
            print("非法参数:", x)
            quit()

         i += 1

   files = find_file_list(files_pattern)
   if len(files) == 0:
      print("No file!")
      quit()

   process_file_update(files)

if __name__ == '__main__':
   main()
