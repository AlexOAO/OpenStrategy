#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段3：X2計算（20日累積週轉率）
計算T-23到T-4的20個交易日累積週轉率
符合claude.md日頻事件研究法架構
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 檢測專案根目錄
def get_project_root():
    """取得專案根目錄"""
    current = Path.cwd()
    if current.name == 'src':
        return current.parent
    return current

PROJECT_ROOT = get_project_root()


class TurnoverCalculator:
    """20日累積週轉率計算器（X2變數）"""

    def __init__(
        self,
        event_list_path=None,
        car_data_path=None,
        tool_aprcd1=None
    ):
        """
        初始化X2計算器

        Parameters:
        -----------
        event_list_path : str
            事件列表檔案路徑（備用）
        car_data_path : str
            CAR資料檔案路徑（優先使用）
        tool_aprcd1 : str
            TEJ日成交資料工具
        """
        self.event_list_path = event_list_path or (PROJECT_ROOT / 'data/processed/event_list.csv')
        self.car_data_path = car_data_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        self.tool_aprcd1 = tool_aprcd1 or str(PROJECT_ROOT / 'tej_tool_TWN_APRCD1.py')
        self.events_df = None
        self.x2_results = []

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
            print(f"載入 {len(self.events_df)} 筆成功計算CAR的事件\n")
        else:
            print(f"CAR 資料不存在，從事件列表載入: {self.event_list_path}")
            self.events_df = pd.read_csv(self.event_list_path)
            self.events_df['mdate'] = pd.to_datetime(self.events_df['mdate'])
            print(f"總共 {len(self.events_df)} 筆事件\n")
        
        return self

    def get_daily_turnover(self, coid, event_date):
        """
        取得日成交資料（T-23到T-4，共20個交易日）

        Parameters:
        -----------
        coid : str
            股票代號
        event_date : datetime
            事件日期（T=0）

        Returns:
        --------
        pd.DataFrame : 包含成交量與流通股數的DataFrame
        """
        try:
            # 🚀 優化：使用365天快取視窗（減少重複API呼叫）
            start_date = (event_date - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = (event_date - timedelta(days=3)).strftime('%Y-%m-%d')

            # 檢查是否已有現成檔案
            output_dir = PROJECT_ROOT / 'output_aprcd1'
            output_dir.mkdir(exist_ok=True)
            existing_files = list(output_dir.glob(f'APRCD1_{coid}_*.csv'))

            # 檢查 cache 檔案是否包含足夠的日期範圍
            use_cache = False
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                try:
                    df_check = pd.read_csv(latest_file)
                    if len(df_check) > 0 and 'mdate' in df_check.columns:
                        df_check['mdate'] = pd.to_datetime(df_check['mdate'])
                        cache_start = df_check['mdate'].min()
                        cache_end = df_check['mdate'].max()
                        required_start = pd.Timestamp(start_date)
                        required_end = pd.Timestamp(end_date)
                        
                        if cache_start <= required_start and cache_end >= required_end:
                            use_cache = True
                except:
                    use_cache = False

            if not use_cache:
                print(f"  [週轉率] 呼叫TEJ API: {coid}")
                cmd = [
                    'python3', self.tool_aprcd1,
                    '--code', str(coid),
                    '--start', start_date,
                    '--end', end_date
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)
                existing_files = list(output_dir.glob(f'APRCD1_{coid}_*.csv'))
            else:
                print(f"  [週轉率] 使用現有檔案 (涵蓋範圍完整)")

            if not existing_files:
                return None

            # 讀取日成交資料
            latest_file = max(existing_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            # 檢查檔案是否為空
            if len(df) == 0:
                print(f"  [週轉率] 警告：檔案為空")
                return None
                
            df['mdate'] = pd.to_datetime(df['mdate'])

            return df

        except Exception as e:
            print(f"  [週轉率] 錯誤：無法取得{coid}的成交資料")
            print(f"  [週轉率] 詳細錯誤: {str(e)}")
            return None

    def calculate_x2_for_event(self, coid, event_date):
        """
        計算單一事件的X2（20日累積週轉率）

        根據claude.md：
        - X2 = Σ(日成交量/流通在外股數) from T-23 to T-4
        - 共20個交易日（注意：是交易日而非日曆日）
        - 單位：百分比

        Parameters:
        -----------
        coid : str
            股票代號
        event_date : datetime
            事件日期

        Returns:
        --------
        dict : X2結果
        """
        # 取得日成交資料
        turnover_df = self.get_daily_turnover(coid, event_date)
        if turnover_df is None or len(turnover_df) == 0:
            return None

        # 排序並定位事件日
        turnover_df = turnover_df.sort_values('mdate').reset_index(drop=True)

        # 找到事件日或最接近的交易日
        event_mask = turnover_df['mdate'] == event_date
        if event_mask.any():
            event_idx = turnover_df[event_mask].index[0]
        else:
            # 尋找最接近的交易日
            turnover_df['date_diff'] = abs((turnover_df['mdate'] - event_date).dt.days)
            event_idx = turnover_df['date_diff'].idxmin()
            turnover_df = turnover_df.drop('date_diff', axis=1)

        # 定位時間窗期（T-23到T-4，共20個交易日）
        # T-23到T-4: 從event_idx-23到event_idx-4（共20個交易日）
        window_start = max(0, event_idx - 23)
        window_end = max(0, event_idx - 3)  # T-4是event_idx-4，但iloc是左閉右開，所以是-3

        # 取得窗期資料
        window_df = turnover_df.iloc[window_start:window_end].copy()

        # 需要至少15個交易日
        if len(window_df) < 15:
            print(f"  [X2] 交易日數不足: {len(window_df)} < 15")
            return None

        # 計算每日週轉率
        # APRCD1直接提供turnover欄位（已是小數形式，例如0.0427代表4.27%）
        if 'turnover' in window_df.columns:
            # turnover已經是ratio形式，乘以100轉為百分比
            window_df['daily_turnover'] = window_df['turnover'] * 100
        elif 'volume' in window_df.columns and 'outstanding' in window_df.columns:
            # 備用計算：成交量/流通股數
            window_df['daily_turnover'] = (window_df['volume'] / window_df['outstanding']) * 100
        else:
            print(f"  [週轉率] 缺少必要欄位，可用欄位: {list(window_df.columns)}")
            return None

        # 計算20日累積週轉率
        x2_cumulative_turnover = window_df['daily_turnover'].sum()

        return {
            'coid': coid,
            'event_date': event_date,
            'X2_cumulative_turnover_20d': x2_cumulative_turnover,
            'n_days': len(window_df)
        }

    def process_events(self, sample_size=None, start_date='2020-01-01', end_date='2025-12-31'):
        """
        批次處理所有事件

        Parameters:
        -----------
        sample_size : int, optional
            樣本數量限制（用於測試）
        start_date : str
            事件日期起始範圍（YYYY-MM-DD）
        end_date : str
            事件日期結束範圍（YYYY-MM-DD）
        """
        print("="*80)
        print("開始計算X2（20日累積週轉率）")
        print("="*80)

        # 使用參數化的日期範圍過濾
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        events_filtered = self.events_df[
            (self.events_df['mdate'] >= start_ts) & 
            (self.events_df['mdate'] <= end_ts)
        ].copy()
        print(f"過濾至 {start_date} ~ {end_date} 事件: {len(events_filtered)} 筆\n")

        # 統一排序：先按日期、再按股票代號（確保所有階段處理相同順序的事件）
        events_sorted = events_filtered.sort_values(['mdate', 'coid'], ascending=True)

        if sample_size:
            events_to_process = events_sorted.head(sample_size)
            print(f"處理樣本數量: {len(events_to_process)}（從 {start_date} 開始）\n")
        else:
            events_to_process = events_sorted
            print(f"處理全部事件: {len(events_to_process)} 筆\n")

        total = len(events_to_process)

        # 🚀 使用多線程並行處理（加速 3-5 倍）
        max_workers = 6  # 增加到 8 個線程以提升速度
        print(f"🚀 使用 {max_workers} 個線程並行處理\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_event = {}
            for idx, (_, row) in enumerate(events_to_process.iterrows(), 1):
                coid = row['coid']
                event_date = row['mdate']
                
                future = executor.submit(self.calculate_x2_for_event, coid, event_date)
                future_to_event[future] = (idx, coid, event_date)
            
            # 收集結果（按完成順序）
            completed = 0
            for future in as_completed(future_to_event):
                idx, coid, event_date = future_to_event[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.x2_results.append(result)
                        print(f"✓ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - X2: {result['X2_cumulative_turnover_20d']:.2f}%")
                    else:
                        print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 無法計算X2")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 錯誤: {e}")

        print(f"\n完成！成功計算 {len(self.x2_results)} 筆事件的 X2\n")

    def save_results(self, output_path=None):
        """儲存X2結果"""
        output_path = output_path or (PROJECT_ROOT / 'data/processed/x2_turnover.csv')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.x2_results:
            print("警告：沒有X2結果可儲存")
            return

        df = pd.DataFrame(self.x2_results)
        df.to_csv(output_path, index=False)

        print(f"X2資料已儲存至: {output_path}")
        print(f"共 {len(df)} 筆結果\n")

        # 摘要統計
        print("=== X2摘要統計 ===\n")
        print(f"平均: {df['X2_cumulative_turnover_20d'].mean():.2f}%")
        print(f"中位數: {df['X2_cumulative_turnover_20d'].median():.2f}%")
        print(f"標準差: {df['X2_cumulative_turnover_20d'].std():.2f}%")
        print(f"最小值: {df['X2_cumulative_turnover_20d'].min():.2f}%")
        print(f"最大值: {df['X2_cumulative_turnover_20d'].max():.2f}%")


def main():
    """主程式"""
    print("="*80)
    print("階段3：X2計算（20日累積週轉率，T-23到T-4）")
    print("="*80)
    print()

    calculator = TurnoverCalculator()
    calculator.load_events()
    calculator.process_events(sample_size=10)  # 測試：先處理10筆
    calculator.save_results()

    print("階段3完成！\n")


if __name__ == '__main__':
    main()
