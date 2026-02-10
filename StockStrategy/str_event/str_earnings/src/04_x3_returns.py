#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段4：X3計算（10日累積報酬率）
計算T-13到T-4的10個交易日累積報酬率
符合claude.md日頻事件研究法架構
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_project_root():
    """取得專案根目錄"""
    current = Path.cwd()
    if current.name == 'src':
        return current.parent
    return current

PROJECT_ROOT = get_project_root()


class ReturnsCalculator:
    """10日累積報酬率計算器（X3變數）"""

    def __init__(
        self,
        event_list_path=None,
        car_data_path=None,
        tool_returns=None
    ):
        """初始化X3計算器"""
        self.event_list_path = event_list_path or (PROJECT_ROOT / 'data/processed/event_list.csv')
        self.car_data_path = car_data_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        # 改用 ABETAD1（與階段2保持一致）
        self.tool_returns = tool_returns or str(PROJECT_ROOT / 'tej_tool_TWN_ABETAD1.py')
        self.events_df = None
        self.x3_results = []

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

    def get_daily_returns(self, coid, event_date):
        """取得日報酬率資料（T-13到T-4）- 使用 ABETAD1"""
        try:
            # 🚀 優化：使用365天快取視窗（減少重複API呼叫）
            start_date = (event_date - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = (event_date - timedelta(days=3)).strftime('%Y-%m-%d')

            # 使用 ABETAD1 輸出目錄
            output_dir = PROJECT_ROOT / 'output_abetad1'
            output_dir.mkdir(exist_ok=True)

            # 檢查是否已有檔案
            existing_files = list(output_dir.glob(f'ABETAD1_{coid}_*.csv'))

            # 檢查 cache 檔案是否包含足夠的日期範圍
            use_cache = False
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                try:
                    df_check = pd.read_csv(latest_file)
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
                print(f"  [報酬率] 呼叫 TEJ ABETAD1 API: {coid}")
                cmd = ['python3', self.tool_returns, '-c', str(coid), '-start', start_date, '-end', end_date]
                subprocess.run(cmd, capture_output=True, timeout=60)
                existing_files = list(output_dir.glob(f'ABETAD1_{coid}_*.csv'))
            else:
                print(f"  [報酬率] 使用現有檔案")

            if not existing_files:
                return None

            latest_file = max(existing_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            df['mdate'] = pd.to_datetime(df['mdate'])
            return df

        except Exception as e:
            print(f"  [報酬率] 錯誤: {str(e)}")
            return None

    def calculate_x3_for_event(self, coid, event_date):
        """
        計算單一事件的X3（10日累積報酬率）

        根據claude.md：
        - X3 = 累積報酬率 from T-13 to T-4
        - 使用連乘法：[(1+R₁) × (1+R₂) × ... × (1+R₁₀)] - 1
        - 共10個交易日（注意：是交易日而非日曆日）
        - 單位：百分比
        """
        returns_df = self.get_daily_returns(coid, event_date)
        if returns_df is None or len(returns_df) == 0:
            return None

        # 排序並定位事件日
        returns_df = returns_df.sort_values('mdate').reset_index(drop=True)

        # 找到事件日或最接近的交易日
        event_mask = returns_df['mdate'] == event_date
        if event_mask.any():
            event_idx = returns_df[event_mask].index[0]
        else:
            # 尋找最接近的交易日
            returns_df['date_diff'] = abs((returns_df['mdate'] - event_date).dt.days)
            event_idx = returns_df['date_diff'].idxmin()
            returns_df = returns_df.drop('date_diff', axis=1)

        # 定位時間窗期（T-13到T-4，共10個交易日）
        window_start = max(0, event_idx - 13)
        window_end = max(0, event_idx - 3)  # T-4是event_idx-4，iloc左閉右開所以-3

        window_df = returns_df.iloc[window_start:window_end].copy()

        if len(window_df) < 8:  # 至少需要8個交易日
            print(f"  [X3] 交易日數不足: {len(window_df)} < 8")
            return None

        # 使用roi欄位（ABETAD1日報酬率%）
        if 'roi' in window_df.columns:
            # ★★★ 修正：使用連乘法計算累積報酬率 ★★★
            # 累積報酬率 = [(1+R1) × (1+R2) × ... × (1+R10)] - 1
            # roi 是百分比，需要先除以100轉換為小數
            daily_returns = window_df['roi'] / 100.0
            
            # 連乘：(1 + R1) × (1 + R2) × ... × (1 + R10)
            cumulative_multiplier = (1 + daily_returns).prod()
            
            # 減1得到累積報酬率，再轉回百分比
            x3_cumulative_return = (cumulative_multiplier - 1) * 100.0
        else:
            print(f"  [報酬率] 缺少roi欄位")
            return None

        return {
            'coid': coid,
            'event_date': event_date,
            'X3_cumulative_return_10d': x3_cumulative_return,
            'n_days': len(window_df)
        }

    def process_events(self, sample_size=None, start_date='2020-01-01', end_date='2025-12-31'):
        """批次處理所有事件"""
        print("="*80)
        print("開始計算X3（10日累積報酬率）")
        print("="*80)

        # 使用參數化的日期範圍過濾
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        events_filtered = self.events_df[
            (self.events_df['mdate'] >= start_ts) &
            (self.events_df['mdate'] <= end_ts)
        ].copy()
        # 統一排序：先按日期、再按股票代號（確保所有階段處理相同順序的事件）
        events_sorted = events_filtered.sort_values(['mdate', 'coid'], ascending=True)

        if sample_size:
            events_to_process = events_sorted.head(sample_size)
        else:
            events_to_process = events_sorted

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
                
                future = executor.submit(self.calculate_x3_for_event, coid, event_date)
                future_to_event[future] = (idx, coid, event_date)
            
            # 收集結果（按完成順序）
            completed = 0
            for future in as_completed(future_to_event):
                idx, coid, event_date = future_to_event[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.x3_results.append(result)
                        print(f"✓ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - X3: {result['X3_cumulative_return_10d']:.2f}%")
                    else:
                        print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 無法計算X3")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 錯誤: {e}")

        print(f"\n完成！成功計算 {len(self.x3_results)} 筆\n")

    def save_results(self, output_path=None):
        """儲存X3結果"""
        output_path = output_path or (PROJECT_ROOT / 'data/processed/x3_returns.csv')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.x3_results:
            print("警告：沒有X3結果")
            return

        df = pd.DataFrame(self.x3_results)
        df.to_csv(output_path, index=False)
        print(f"X3資料已儲存至: {output_path}")
        print(f"平均X3: {df['X3_cumulative_return_10d'].mean():.2f}%")


def main():
    """主程式"""
    print("="*80)
    print("階段4：X3計算（10日累積報酬率）")
    print("="*80)

    calculator = ReturnsCalculator()
    calculator.load_events()
    calculator.process_events(sample_size=10)
    calculator.save_results()
    print("階段4完成！\n")


if __name__ == '__main__':
    main()
