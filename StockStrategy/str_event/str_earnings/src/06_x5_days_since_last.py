#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段6：X5計算（距離前次法說會天數的對數轉換）
計算本次法說會距離前次法說會的天數，並進行對數轉換
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    current = Path.cwd()
    return current.parent if current.name == 'src' else current

PROJECT_ROOT = get_project_root()


class DaysSinceLastCalculator:
    def __init__(self, event_list_path=None, car_data_path=None):
        self.event_list_path = event_list_path or (PROJECT_ROOT / 'data/processed/event_list.csv')
        self.car_data_path = car_data_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        self.events_df = None
        self.x5_results = []

    def load_events(self):
        """
        載入事件列表，優先使用 car_data.csv（只處理成功計算CAR的事件）
        如果 car_data.csv 不存在，則使用 event_list.csv
        """
        if self.car_data_path.exists():
            print(f"從 CAR 資料載入事件: {self.car_data_path}")
            self.events_df = pd.read_csv(self.car_data_path)
            # car_data.csv 使用 event_date 欄位，需統一為 mdate
            if 'event_date' in self.events_df.columns:
                self.events_df['mdate'] = pd.to_datetime(self.events_df['event_date'])
            elif 'mdate' in self.events_df.columns:
                self.events_df['mdate'] = pd.to_datetime(self.events_df['mdate'])
            
            # 只保留必要欄位
            self.events_df = self.events_df[['coid', 'mdate']].copy()
            print(f"載入 {len(self.events_df)} 筆成功計算CAR的事件")
        else:
            print(f"CAR 資料不存在，從事件列表載入: {self.event_list_path}")
            self.events_df = pd.read_csv(self.event_list_path)
            self.events_df['mdate'] = pd.to_datetime(self.events_df['mdate'])
            print(f"載入 {len(self.events_df)} 筆事件")
        
        return self

    def calculate_x5_for_all_events(self, start_date='2020-01-01', end_date='2025-12-31'):
        """
        計算X5：距離前次法說會天數的對數轉換

        對每個股票按時間排序，計算當前事件與前一次事件的天數差
        然後進行對數轉換：log(天數 + 1)
        第一次事件X5 = log(中位數 + 1)
        
        Parameters:
        -----------
        start_date : str
            事件日期起始範圍（YYYY-MM-DD）
        end_date : str
            事件日期結束範圍（YYYY-MM-DD）
        """
        print("="*80)
        print("階段6：X5計算（距離前次法說會天數的對數轉換）")
        print("="*80)

        # 按股票和日期排序（使用所有事件來正確計算間隔）
        df_sorted = self.events_df.sort_values(['coid', 'mdate']).copy()

        # 計算同一股票前一次事件的日期
        df_sorted['prev_mdate'] = df_sorted.groupby('coid')['mdate'].shift(1)

        # 計算天數差
        df_sorted['days_since_last'] = (df_sorted['mdate'] - df_sorted['prev_mdate']).dt.days

        # 第一次事件填中位數
        median_days = df_sorted['days_since_last'].median()
        df_sorted['days_since_last'].fillna(median_days, inplace=True)

        # 進行對數轉換：log(天數 + 1)
        df_sorted['X5_log_days_since_last'] = np.log(df_sorted['days_since_last'] + 1)

        # 過濾到指定日期範圍
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        df_filtered = df_sorted[
            (df_sorted['mdate'] >= start_ts) & 
            (df_sorted['mdate'] <= end_ts)
        ].copy()

        self.x5_results = df_filtered[['coid', 'mdate', 'X5_log_days_since_last', 'days_since_last']].to_dict('records')

        print(f"完成 {len(self.x5_results)} 筆（日期範圍：{start_date} ~ {end_date}）")
        print(f"平均間隔天數: {df_filtered['days_since_last'].mean():.1f} 天")
        print(f"對數轉換後平均值: {df_filtered['X5_log_days_since_last'].mean():.3f}\n")

    def save_results(self, output_path=None):
        output_path = output_path or (PROJECT_ROOT / 'data/processed/x5_days_since_last.csv')
        if not self.x5_results:
            print("無X5結果")
            return
        df = pd.DataFrame(self.x5_results)
        df.to_csv(output_path, index=False)
        print(f"X5已儲存: {output_path}")


def main():
    calculator = DaysSinceLastCalculator()
    calculator.load_events()
    calculator.calculate_x5_for_all_events()
    calculator.save_results()


if __name__ == '__main__':
    main()
