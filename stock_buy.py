import numpy as np
import math, sys, os
import tushare as ts
import pandas as pd
import datetime as dt
from pandas import read_csv
from collections import *
from functools import *

verbose = False
tran_verbose = False
output_year_profits = False
t0_tran = False
tran_strategy = [0]
find_avg_line_by_year = False
find_avg_line_years = 1
find_avg_line_verbose = False
find_avg_line_by_code = False
test_for_best_avg_line = False
test_for_best_avg_years = 1
sort_by_profit = False
no_any_sort = False
min_win_ratio = 0
max_win_ratio = 100
is_white_horse = False
kline_type = 'D'
TRAN_COST = 0.0015

def ftos(fs):
   if type(fs) == list:
      return ' '.join([('%.3f' % f) for f in fs])
   else:
      return '%.3f' % fs

def ftos2(fs):
   return '%+.3f' % fs

def itos(v):
   return '% 4d' % v

def right_padding(s, cnt):
   return ('{:<' + str(cnt) + 's}').format(s)

def load_stock_data_from_file(file):
   failed = False
   try:
      # 从整理好的文件中获取数据
      df = read_csv(file, engine='python')
      if len(df.columns) != 5:
         failed = True
   except:
      failed = True

   if failed:
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
      df = read_csv(file, usecols=[0,1,2,3,4], sep='\s+', engine='python')

   df.columns = ['date', 'open', 'high', 'low', 'close']
   return df

def load_stock_data_from_network(code, kline_type):
   df = ts.get_k_data(code, '19900101', '', ktype=kline_type)
   if len(df) > 0:
      # 把收盘价放最低价后面，保持和同花顺一致顺序
      cols = list(df)
      cols.insert(4, cols.pop(2))
      df = df.loc[:, cols]

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

def data_frame_to_dataset(df, date_start='', date_end=''):
   if len(df) == 0:
      return [], [], [], []

   if date_start != '':
      df = df[df[df.columns[0]] >= date_start]

   if date_end != '':
      df = df[df[df.columns[0]] < date_end]

   ds = df.values
   return ds[:, 0], ds[:, 1], ds[:, 2], ds[:, 3], ds[:, 4]

def load_stock_dataset(file_or_code, date_start='', date_end=''):
   df = load_stock_data_frame(file_or_code, date_start, date_end)
   return data_frame_to_dataset(df)

def get_stock_date(ds):
   return ds[0]

def get_stock_open(ds):
   return ds[1]

def get_stock_max(ds):
   return ds[2]

def get_stock_min(ds):
   return ds[3]

def get_stock_close(ds):
   return ds[4]

def get_date_time(ds_date, i):
   if ds_date[0] == ds_date[1]:
      j = 1
      while ds_date[i-j] == ds_date[i]:
         j += 1

      min = 570
      if ds_date[0] == ds_date[240//1 - 2]:
      	min += j
      elif ds_date[0] == ds_date[240 // 5 -2]:
      	min += j * 5
      elif ds_date[0] == ds_date[240 // 15 -2]:
         min += j * 15
      elif ds_date[0] == ds_date[240 // 30 -2]:
         min += j * 30
      elif ds_date[0] == ds_date[240 // 60 -2]:
         min += j * 60
      else:
         min += j * 120

      if min > 570+120:
         min += 90

      time = ' ' + ('%02d' % (min // 60)) + ":" + ('%02d' % (min % 60))
   else:
      time = ''

   return ds_date[i][0:10] + time

def get_tran_year(ds_date, i):
   return int(ds_date[i][0:4])

# 获取当前值往前n个数据的最大值
def maxn(a, i, n):
   if i > 0:
      if i+1 >= n:
         cnt = n
         s = i+1-n
      else:
         cnt = i+1
         s = 0

      m = 0.0
      while s <= i:
         v = a[s]
         if m < v:
            m = v
         s += 1
      return m
   else:
      return 0.0

# 获取当前值往前n个数据的最小值
def minn(a, i, n):
   if i > 0 and n > 0:
      s = max(0, i+1-n)
      return min(a[s : i+1])
   else:
      return 0.0

# 获取当前值往前n个数据的均值
def avg(a, i, n):
   if i > 0:
      if i+1 >= n:
         cnt = n
         s = i+1-n
      else:
         cnt = i+1
         s = 0

      m = 0.0
      while s <= i:
         m += a[s]
         s += 1
      return m/cnt
   else:
      return 0.0

# 当前值是否大于第前n个数据的值
# 在图像上表现均线向上
def m_line_up(a, i, n):
   return i >= n and a[i] > a[i-n]

# 多根均线都向上
def m_line_up_all(a, i, ns):
   ll = len(ns)
   if ll == 1:
      return m_line_up(a, i, ns[0])
   elif ll == 2:
      return m_line_up(a, i, ns[1]) and m_line_up(a, i, ns[0])
   elif ll == 3:
      return m_line_up(a, i, ns[2]) and m_line_up(a, i, ns[1]) and m_line_up(a, i, ns[0])
   else:
      return all([m_line_up(a, i, n) for n in ns])

# 一根均线向下
def m_line_down(a, i, n):
   return i >= n and a[i] < a[i-n]

# 一根均线向下
def m_line_down_all(a, i, ns):
   ll = len(ns)
   if ll == 1:
      return m_line_down(a, i, ns[0])
   elif ll == 2:
      return m_line_down(a, i, ns[1]) and m_line_down(a, i, ns[1])
   elif ll == 3:
      return m_line_down(a, i, ns[2]) and m_line_down(a, i, ns[1]) and m_line_down(a, i, ns[0])
   else:
      return all([m_line_down(a, i, n) for n in ns])

MAX_LOSS_DAY = 0.08
MAX_LOSS_MIN = 0.00
MAX_LOSS = 0.00

def stop_loss(buy_price, open_min_price):
   return MAX_LOSS > 0 and 1 - open_min_price/buy_price >= MAX_LOSS

# 分钟线，较晚进，较早出
# 3根均线都向上
def buy1(a, i, mm):
   return m_line_up_all(a, i, mm)

# 均线三向下
def sell1(a, i, mm):
   return m_line_down(a, i, mm[2])

# 日线，尽可能早进，尽可能晚出
# 股价站上均线一
def buy2(a, i, mm):
   return a[i] > avg(a, i, mm[0])

# 均线二向下：这是确保买入后不会立刻卖出，并且不会在上涨的时候卖出
# 并且
# 均线三向下或者股价三天在均线三下
def sell2(a, i, mm):
   return m_line_down(a, i, mm[1]) and (m_line_down(a, i, mm[2]) or max(a[i], a[i-1], a[i-2]) < avg(a, i, mm[2]))

def buy3(a, i, mm):
   return m_line_up_all(a, i, mm[0:2])

def sell3(a, i, mm):
   return m_line_down_all(a, i, mm[1:3])

def buy4(a, i, mm):
   return m_line_up_all(a, i, mm[0])

def sell4(a, i, mm):
   return m_line_down_all(a, i, mm)

def buy5(a, i, mm):
   pass

def sell5(a, i, mm):
   pass

def buy6(a, i, mm):
   pass

def sell6(a, i, mm):
   pass

def get_tran_strategy(cycle):
   return [get_buy_tran_strategy(tran_strategy[0], cycle),
           get_sell_tran_strategy(tran_strategy[1], cycle)]

def get_buy_tran_strategy(x, cycle):
   v = [buy1, buy2, buy3, buy4, buy5, buy6]
   if x >= 1 and x <= len(v):
      return v[x-1]
   else:
      return buy1 if cycle < 240 else buy2

def get_sell_tran_strategy(x, cycle):
   v = [sell1, sell2, sell3, sell4, sell5, sell6]
   if x >= 1 and x <= len(v):
      return v[x-1]
   else:
      return sell1 if cycle < 240 else sell2

def tran(ds, mm, buy_sell_strategy):
   buy, sell = buy_sell_strategy
   ds_date = get_stock_date(ds)
   ds_open = get_stock_open(ds)
   ds_max = get_stock_max(ds)
   ds_min = get_stock_min(ds)
   ds_close = get_stock_close(ds).tolist()

   total_value = 1.0
   buy_price = 0.0
   buy_index = 0
   buy_time = ''
   buy_date = ''
   buying = False

   last_tran_year = 0
   last_tran_year_stock_open_price = 0.0
   last_tran_year_value = 0.0
   year_profits = []

   loss_count, win_count = 0, 0
   total_blank_profit, blank_profit = 0.0, 0.0
   last_sell_price = 0.0

   data_len = len(ds_date)
   for i in range(1, data_len):
      # 处理每年的收益
      if output_year_profits:
         tran_year = get_tran_year(ds_date, i)
         if tran_year != last_tran_year or i == data_len - 1:
            next_year_start_value = total_value
            if last_tran_year > 0:
               # 把收益情况记录在year_profits中
               if buying:
                  may_profit = (ds_close[i] - buy_price) / buy_price
                  next_year_start_value = total_value * (1 + may_profit - TRAN_COST)

               profit = (next_year_start_value - last_tran_year_value) / last_tran_year_value
               total_profit = next_year_start_value - 1.0
               stock_change = (ds_close[i-1] - last_tran_year_stock_open_price) / last_tran_year_stock_open_price
               total_stock_change = (ds_close[i-1] - ds_open[0]) / ds_open[0]

               year_profits += [(last_tran_year, profit, total_profit, stock_change, total_stock_change)]

            # 切换交易年
            last_tran_year = tran_year
            last_tran_year_stock_open_price = ds_open[i]
            last_tran_year_value = next_year_start_value

      # 处理买卖点
      if not buying and buy(ds_close, i, mm):
         buying = True
         buy_price = ds_close[i]
         buy_index = i
         buy_date = ds_date[i]

         if tran_verbose and last_sell_price > 0.0:
            blank_profit = (last_sell_price - buy_price) / last_sell_price
            total_blank_profit += blank_profit

      elif buying:
         try_sell = True

         if MAX_LOSS > 0 and stop_loss(buy_price, ds_open[i]):
            # 如果开盘价达到止损，那么挂成本价，如果最高价达到该价格
            # 那么成功卖出，否则以收盘价卖出
            # 如果盘中达到止损价，那么以止损价卖出
            if ds_max[i] > buy_price * (1 + TRAN_COST):
               profit = 0.0
               last_sell_price = buy_price * (1 + TRAN_COST)
            else:
               last_sell_price = ds_close[i]
               profit = last_sell_price / buy_price - 1.0 - TRAN_COST
         elif MAX_LOSS > 0 and stop_loss(buy_price, ds_min[i]):
            profit = -MAX_LOSS-TRAN_COST
            last_sell_price = buy_price * (1 - MAX_LOSS)
         elif i == data_len-1 or ((t0_tran or ds_date[i] != buy_date) and sell(ds_close, i, mm)):
            last_sell_price = ds_close[i]
            profit = last_sell_price / buy_price - 1.0 - TRAN_COST
         else:
            try_sell = False

         if try_sell:
            total_value *= (1 + profit)

            if profit >= 0:
               win_count += 1
            else:
               loss_count += 1

            buying = False
            if tran_verbose:
               total_stock_value = ds_close[i]/ds_open[0]
               buy_time = get_date_time(ds_date, buy_index)
               print(itos(win_count + loss_count), buy_time, get_date_time(ds_date, i), ftos(buy_price), ftos(ds_close[i]),
                     ftos2(profit * 100) + '%', ftos2(blank_profit * 100) + '%', ftos(total_value),
                     ftos(total_stock_value)
                     )

   if tran_verbose:
      print('')

   if win_count + loss_count == 0:
      win = 0.0
   else:
      win = win_count / (win_count + loss_count) * 100

   return total_value, win, win_count + loss_count, year_profits

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

def stock_name(f):
   return f.partition('.')[0]

def extract_cycle_from_name(fn):
   if fn.find("01MIN") >= 0 or fn.find("-1M-") >= 0:
      return 1
   elif fn.find("05MIN") >= 0 or fn.find("-5M-") >= 0:
      return 5
   elif fn.find("15MIN") >= 0 or fn.find("-QH-") >= 0:
      return 15
   elif fn.find("30MIN") >= 0 or fn.find("-HH-") >= 0:
      return 30
   elif fn.find("1HOUR") >= 0 or fn.find("60MIN") >= 0 or fn.find("-1H-") >= 0:
      return 60
   elif fn.find("2HOUR") >= 0 or fn.find("120MIN") >= 0 or fn.find("-2H-") >= 0:
      return 120
   else:
      return 240

warning_titles = {}
def process_tran(ds, mm, buy_sell_strategy, title):
   if verbose:
      print('Processing:', title, '均线:', mm)

   if len(ds[0]) < 50:
      if not title in warning_titles:
         print('行情数据记录条数 [' + str(len(ds[0])) + '] 小于100')
         warning_titles[title] = title
      return 1.0, 0.0, 0.0, mm, title, 0, []

   vs, win, tran_counts, year_profits = tran(ds, mm, buy_sell_strategy)
   stock_price_change = (ds[4][-1] - ds[1][0]) / ds[1][0]
   return vs, win, stock_price_change, mm, title, tran_counts, year_profits

def process_tran_for_one(ds, file_or_code, mm = []):
   cycles = {
               1:  (12*30, 25*30, 20*30),
               5:  (12*6, 25*6, 20*6),
               15: (12*2, 25*2, 20*2),
               30: (12, 25, 20),
               60: (6, 12, 10),
               120:(3, 6, 5),
               240:(5, 3, 20),
            }

   if is_file_name(file_or_code):
      fn = file_or_code.upper()
      cycle = extract_cycle_from_name(fn)
   else:
      fn = file_or_code
      if kline_type.isdigit():
         cycle = int(kline_type)
      else:
         cycle = 240

   if len(mm) == 0:
      mm = cycles[cycle]

   global MAX_LOSS
   MAX_LOSS = MAX_LOSS_MIN if cycle < 240 else MAX_LOSS_DAY
   buy_sell_strategy = get_tran_strategy(cycle)

   return process_tran(ds, mm, buy_sell_strategy, stock_name(fn))

def process_tran_for_file_or_network(file_or_codes, mms = [], date_start='', date_end=''):
   result = []
   for file_or_code in file_or_codes:
      ds = load_stock_dataset(file_or_code, date_start, date_end)
      if len(ds[0]) >= 50:
         for mm in mms:
            result += [process_tran_for_one(ds, file_or_code, mm)]
   if not no_any_sort:
      if sort_by_profit:
         result = sorted(result, key=lambda x: x[0]-x[2], reverse=True)
      else:
         result = sorted(result, key=lambda x: x[0], reverse=True)
   output_result(result)

def get_best_avg_lines(result, cnt):
   min_ratio = min_win_ratio
   max_ratio = max_win_ratio
   while True:
      best = result
      if min_ratio > 0:
         best = [x for x in best if x[1] >= min_ratio]

      if max_ratio < 100:
         best = [x for x in best if x[1] <= max_ratio]

      if len(best) >= cnt or len(best) == len(result):
         break
      else:
         min_ratio -= 2
         max_ratio += 2

   best = sorted(best, reverse=True)
   return best[0:min(cnt, len(best))]

def add_year(file_or_code, year_start, year_end=''):
   if is_file_name(file_or_code):
      v = file_or_code[0:-4] + '-' + year_start
      if year_end != '' and int(year_end) - int(year_start) > 1:
         v += '~' + year_end
      v += '.TXT'
   else:
      v = file_or_code + '-' + year_start
      if year_end != '' and int(year_end) - int(year_start) > 1:
         v += '~' + year_end
   return v

def output_best_avg_lines(desc, avg_lines):
   print(desc + ' 最佳组合:')
   for mm, vs, win, tran_counts in avg_lines:
      print('   ', mm, ftos(vs), str(tran_counts) + '(' + ftos(win/tran_counts*100) + '%)')
   print('')

def collect_tran_result(ds, file_or_code, mms):
   result = []
   for mm in mms:
      result += [process_tran_for_one(ds, file_or_code, mm)]
   return result

def sort_avg_lines(a):
   if len(a) <= 1:
      return a

   result = sorted(a, key=lambda x: x[1], reverse=True)
   result = [result[0]] + sorted(result[1:], key=lambda x: x[2]/x[3], reverse=True)
   return result

def merge_same_and_sort(a, b):
   result = []
   for mm, vs, win, tran_counts in a:
      for mm2, vs2, win2, tran_counts2 in b:
         if mm == mm2:
            result.append((mm, vs*vs2, win+win2, tran_counts+tran_counts2))
            break
   return sort_avg_lines(result)

def find_same_avg_line_by_year(df, file_or_code, mms, cnt, date_start, date_end):
   same_avg_lines = []
   first = True
   dt_start = max(df.iloc[0,0][0:4], date_start)
   dt_last = min(df.iloc[-1,0], date_end) if date_end != '' else df.iloc[-1,0]

   while dt_start < dt_last:
      dt_next = str(int(dt_start) + find_avg_line_years)
      ds = data_frame_to_dataset(df, dt_start, dt_next)
      result = []
      if len(ds[0]) >= 50:
         name = add_year(file_or_code, dt_start, min(dt_next, dt_last[0:4]))
         for mm in mms:
            result += [process_tran_for_one(ds, name, mm)]
         result = get_best_avg_lines(result, max(cnt, 50))

         if find_avg_line_verbose:
            output_result(result[:cnt])

         cur_avg_lines = [(x[3],x[0],int(x[1]*x[5]/100),x[5]) for x in result]
         if first:
            first = False
            same_avg_lines = sort_avg_lines(cur_avg_lines)
         else:
            same_avg_lines = merge_same_and_sort(same_avg_lines, cur_avg_lines)
            if find_avg_line_verbose and len(same_avg_lines) > 0:
               output_best_avg_lines(stock_name(name), same_avg_lines)

      dt_start = dt_next

   return same_avg_lines

def find_best_avg_line_by_year(file_or_code, mms, cnt, date_start, date_end):
   df = load_stock_data_frame(file_or_code)
   same_avg_lines = find_same_avg_line_by_year(df, file_or_code, mms, cnt, date_start, date_end)
   if len(same_avg_lines) > 15:
      same_avg_lines = same_avg_lines[0:15]

   if not find_avg_line_verbose and len(same_avg_lines) > 0:
      output_best_avg_lines(stock_name(file_or_code), same_avg_lines)

   if date_end == '':
      print('没有设置验证开始时间(回测结束时间[-te选项]), 跳过回测')
      return
   elif len(same_avg_lines):
      print('最佳组合为空, 跳过回测')
      return

   same_avg_lines = [x[0] for x in same_avg_lines]
   if test_for_best_avg_line:
      result = []

      # 再测后面的数据
      result = []
      dt_start = date_end
      dt_end = str(int(dt_start[0:4]) + test_for_best_avg_years)
      ds = data_frame_to_dataset(df, dt_start, dt_end)
      if len(ds[0]) >= 50:
         dt_end = ds[0][-1][0:4]
         print('验证段数据的回测:')
         for mm in same_avg_lines:
            result += [process_tran_for_one(ds, add_year(file_or_code, dt_start[0:4], dt_end), mm)]
         output_result(result)

         # 再算这段时间最好的
         print('验证段数据的最佳组合:')
         result = []
         for mm in mms:
            result += [process_tran_for_one(ds, add_year(file_or_code, dt_start[0:4], dt_end), mm)]
         result = get_best_avg_lines(result, min(20, cnt))
         output_result(result)

def find_best_avg_lines_for_file_or_network(file_or_codes, mms, cnt=10, date_start='', date_end=''):
   if find_avg_line_by_year:
      for file_or_code in file_or_codes:
         find_best_avg_line_by_year(file_or_code, mms, cnt, date_start, date_end)
   else:
      total_result = []
      total_files = len(file_or_codes)
      num_file = 0
      t1 = dt.datetime.now()
      for file_or_code in file_or_codes:
         result = []
         num_file += 1
         df = load_stock_data_frame(file_or_code, date_start, date_end)
         if len(df) >= 50:
            ds = data_frame_to_dataset(df)
            result += collect_tran_result(ds, file_or_code, mms)

         total_result += result
         if (find_avg_line_by_code or find_avg_line_verbose) and len(result) > 0:
            print('')
            print(stock_name(file_or_code) + '的最佳组合:')
            output_best_avg_lines(result, cnt)
            print('')
            print('当前所有股票的最佳组合:')
            output_best_avg_lines(total_result, cnt)

            t2 = dt.datetime.now()
            print('')
            print('已完成:', str(int(num_file/total_files*100))+'%',
                  '花费时间:', t2-t1,
                  '剩余时间:', (t2-t1)/num_file * (total_files-num_file))

      if not find_avg_line_by_code and not find_avg_line_verbose:
         print('')
         print('所有股票的最佳组合:')
         output_best_avg_lines(total_result, cnt)

def output_best_avg_lines(result, cnt):
   if len(result) > 0:
      result_df = result_to_dataframe(result)
      #result_df.to_csv('result.csv', index=False)
      result_df["price_change"] = 1+result_df["price_change"]
      result_df.rename(columns={'mm':'均线', 'value':'净值', 'win':'胜率', 'price_change':'股价净值', 'tran_counts':'交易次数'}, inplace=True)
      r = result_df.groupby('均线').mean().sort_values(by='净值', ascending=False).head(cnt)
      r["交易次数"] = r["交易次数"].astype(int)
      pd.set_option('display.precision', 3)
      print(r)

def result_to_dataframe(result):
   result = [x[:-1] for x in result]
   return pd.DataFrame(result, columns=("value", "win", "price_change", "mm", "code", "tran_counts"))

def output_result(tran_result):
   if len(tran_result) == 0:
      return

   total_value = 0.0
   total_gain = 0
   total_counts = 0
   total_stock_price_change = 0.0

   for vs, win, stock_price_change, mms, fn, tran_counts, year_profits in tran_result:
      total_value += vs
      total_gain += win
      total_counts += tran_counts
      total_stock_price_change += stock_price_change
      if output_year_profits:
         print('')
         print('年份 交易收益  股价涨幅  相对收益  总交易收益 总股价涨幅')
         for last_tran_year, profit, total_profit, stock_change, total_stock_change in year_profits:
            print(last_tran_year,
                  right_padding(ftos2(profit*100)+'%', 9),
                  right_padding(ftos2(stock_change*100)+'%', 9),
                  right_padding(ftos2((profit-stock_change)*100)+'%', 9),
                  right_padding(ftos2(total_profit*100)+'%', 10),
                  right_padding(ftos2(total_stock_change*100)+'%', 9),
                 )
         print('')

      print('{:<15s}'.format(str(mms)),
            right_padding(ftos(vs), 8),
            right_padding(ftos2(stock_price_change+1.0), 8),
            right_padding(ftos2((vs-1.0-stock_price_change)), 8),
            '%03d' % tran_counts+'('+ftos(win)+'%)',
            '{:<20s}'.format(stock_name(fn))
           )
   print('')

   avg_value = total_value / len(tran_result)
   avg_stock_value = total_stock_price_change / len(tran_result) + 1.0

   print('交易净值:'+ftos(avg_value),
         '股价净值:'+ftos(avg_stock_value),
         '相对净值:'+ftos(avg_value - avg_stock_value),
         '交易次数:'+ str(total_counts//len(tran_result)) + '(' + ftos(total_gain / len(tran_result))+'%)')
   print('')

def cvt_to_int_list(s):
   def to_list_of(a):
      if a.isdigit():
         return [int(a)]
      else:
         ab = [int(x) for x in a.split('-') if x.isdigit()]
         if len(ab) >= 2:
            return [x for x in range(ab[0], ab[1]+1)]
      return []

   ll = [x for x in s.replace('(', '').replace(')', '').split(',')]
   r = []
   for x in ll:
      r += to_list_of(x)
   return r

def cvt_to_date(s):
   s = s.replace('/', '-')
   if len(s) == 8 and s.isdigit(): # 19980901 --> 1998-09-01
      return s[0:4] + '-' + s[4:6] + '-' + s[6:]
   elif len(s) == 4 and s.isdigit(): # 1998 --> 1998-01-01
      return s + '-01-01'
   elif s.find('-') > 0: # 1998-1-1 --> 1998-01-01
      ls = [int(x) for x in s.split('-') if x.isdigit()]
      if len(ls) == 3:
         return str(ls[0]) + '-%02d' % ls[1] + '-%02d' % ls[2]
   return ''

def generate_mms_all():
   return [(x,y,z) for x in range(5,31,5) for y in range(10,41,5) for z in range(20,61,5)]

def generate_mms_simple():
   return [(x,y,z) for x in range(2,16) for y in range(2,16) for z in range(2,16)]

HelpText = """
+-----------------------------------------------------------------+
|                      均线交易测试工具                           |
|                        Version 1.0                              |
+-----------------------------------------------------------------+

Usage:
  stock_buy [options] stock_code_file_search_str
Options:
  -v                       输出过程信息
  -tv                      输出详细交易细节
  -t0                      是否支持T+0交易
  -cost                    交易成本
  -ts                      数据起始时间(start)
  -te                      数据结束时间(end)
  -yp                      输出每年的收益情况(year profit)
  -w                       交易数据从网络获取
  -wt                      网络交易数据类型:'M','W','D','5','15','30','60'
  -t                       搜索最好的均线组合
  -tm                      搜索更多最好的均线组合
  -ty [n]                  按年搜索最好的均线组合
  -tyt [n]                 回测n年搜索出的均线组合
  -mwr n                   搜索胜率至少多少(min_win_ratio)
  -xwr n                   搜索胜率至多多少(max_win_ratio)
  -tc n                    搜索最好的均线组合, 输出最好的n条
  -tbycode                 按不同的文件或者代码搜索均线组合
  -tran_method n           交易策略
  -xx                      指定第一条均线的搜索范围
  -yy                      指定第二条均线的搜索范围
  -zz                      指定第三条均线的搜索范围
  -xyz n                   指定均线至少是这个倍数
"""

def usage_exit():
   print(HelpText)
   quit()

def main():
   argv = sys.argv[1:]

   if len(argv) <= 0:
      usage_exit()

   global verbose
   global tran_verbose
   global tran_strategy
   global t0_tran
   global find_avg_line_by_year
   global find_avg_line_years
   global find_avg_line_verbose
   global find_avg_line_by_code
   global test_for_best_avg_line
   global test_for_best_avg_years
   global min_win_ratio
   global max_win_ratio
   global kline_type
   global TRAN_COST
   global is_white_horse
   global sort_by_profit
   global no_any_sort
   global output_year_profits

   find_avg_line = False
   find_avg_line_more = False
   output_year_profits = False
   load_data_from_network = False
   tran_date_start = ''
   tran_date_end = ''
   output_avg_lines_count = 10

   mms = []
   code_or_files = []
   xx, yy, zz = [0], [0], [0]
   xyz = 1
   cond = ""

   if True:
      i = 0
      while i < len(argv):
         x = argv[i]

         if x == '-v' or x == '-verbose':
            verbose = True
         elif x == '-tv':
            tran_verbose = True
         elif x == '-t':
            find_avg_line = True
         elif x == '-wh':
            is_white_horse = True
         elif x == '-tm':
            find_avg_line = True
            find_avg_line_more = True
         elif x == '-tbycode':
            find_avg_line = True
            find_avg_line_by_code = True
         elif x == '-tyt':
            test_for_best_avg_line = True
            if i+1 < len(argv) and argv[i+1].isdigit() and len(argv[i+1]) < 3:
               test_for_best_avg_years = int(argv[i+1])
               i += 1
         elif x == '-xyz':
            test_for_best_avg_line = True
            if i+1 < len(argv) and argv[i+1].isdigit():
               xyz = int(argv[i+1])
               i += 1
         elif x == '-cond':
            if i+1 < len(argv):
               if argv[i+1][0] == '"':
                  cond = argv[i+1][1:-1]
               else:
                  cond = argv[i+1]
               i += 1
         elif x == '-tran_method':
            if i+1 < len(argv):
               tran_strategy = cvt_to_int_list(argv[i+1])
               if len(tran_strategy) < 1:
                  print("交易策略必须为数字或者数字列表:", argv[i+1])
                  quit()
               i += 1
         elif x == '-ty' or x == 'ty':
            find_avg_line = True
            find_avg_line_by_year = True
            if i+1 < len(argv) and argv[i+1].isdigit():
               find_avg_line_years = int(argv[i+1])
               i += 1
         elif x == '-cost':
            if i+1 < len(argv):
               if not all([x.isdigit() or x == '.' for x in argv[i+1]]):
                  print("交易成本必须为小数:", argv[i+1])
                  quit()
               TRAN_COST = float(argv[i+1])
               i += 1
         elif x == '-mwr' or x == '-xwr':
            if i+1 < len(argv):
               if not argv[i+1].isdigit():
                  print("胜率必须为整数:", argv[i+1])
                  quit()
               if x == '-mwr':
                  min_win_ratio = int(argv[i+1])
               else:
                  max_win_ratio = int(argv[i+1])
               i += 1
         elif x == '-tc' or x == 'tc':
            find_avg_line = True
            if i+1 < len(argv):
               if not argv[i+1].isdigit():
                  print("输出最好的均线条数必须为整数:", argv[i+1])
                  quit()
               output_avg_lines_count = int(argv[i+1])
               i += 1
         elif x == '-yp':
            output_year_profits = True
         elif x == '-w':
            load_data_from_network = True
         elif x == '-wt':
            load_data_from_network = True
            if i+1 < len(argv):
               if argv[i+1].upper() in ['30','15','5','60','D','W','M']:
                  kline_type = argv[i+1].upper()
               else:
                  print("非法K线周期:", argv[i+1])
                  quit()

               i += 1
         elif x == '-t0':
            t0_tran = True
         elif x == '-sortp':
            sort_by_profit = True
         elif x == '-nosort':
            no_any_sort = True
         elif x == '-m' or x == '-xx' or x == '-yy' or x == '-zz':
            if i+1 < len(argv):
               m = cvt_to_int_list(argv[i+1])
               if x == '-m':
                  if len(m) < 3:
                     print("非法均线组合参数:", argv[i+1])
                     quit()
                  mms.append(m)
               elif x == '-xx':
                  xx = m
               elif x == '-yy':
                  yy = m
               elif x == '-zz':
                  zz = m
               else:
                  pass

               i += 1
         elif x == '-ts' or x == '-te':
            if i+1 < len(argv):
               d = cvt_to_date(argv[i+1])
               if len(d) != 10:
                  print("非法日期参数:", argv[i+1])
                  quit()
               if x == '-ts':
                  tran_date_start = d
               else:
                  tran_date_end = d
               i += 1
         elif x[0] != '-':
            code_or_files.append(x)
         else:
            print("非法参数:", x)
            quit()

         i += 1

   if len(tran_strategy) == 0:
      tran_strategy = [0,0]
   elif len(tran_strategy) == 1:
      tran_strategy = [tran_strategy[0]]*2
   else:
      pass

   if load_data_from_network:
      file_or_codes = [x for x in code_or_files if x.isdigit() and len(x) >= 5 and len(x) <= 6]
      if len(file_or_codes) == 0:
         print("No stock codes provided!")
         quit()
   else:
      ftypes = [x.upper() for x in code_or_files]
      files = find_file_list(ftypes)
      file_or_codes = sorted(files, key=extract_cycle_from_name)

      if len(files) == 0:
         print("No file!")
         quit()

   if find_avg_line:
      if tran_verbose or test_for_best_avg_line:
         find_avg_line_verbose = True
         tran_verbose = False

      if xx == [0] and yy == [0] and zz == [0]:
         mms = generate_mms_all() if find_avg_line_more else generate_mms_simple()
      else:
         mms = [(x,y,z) for x in xx for y in yy for z in zz if x % xyz == 0 and y % xyz == 0 and z % xyz == 0]
         if cond != "":
            mms = [(x,y,z) for x,y,z in mms if eval(cond)]
      t1 = dt.datetime.now()
      find_best_avg_lines_for_file_or_network(file_or_codes, mms, output_avg_lines_count, tran_date_start, tran_date_end)
      t2 = dt.datetime.now()
      print('')
      print('共花费时间:', t2-t1)
   else:
      if len(mms) == 0:
         mms = [[]]
      process_tran_for_file_or_network(file_or_codes, mms, tran_date_start, tran_date_end)

if __name__ == '__main__':
   main()
