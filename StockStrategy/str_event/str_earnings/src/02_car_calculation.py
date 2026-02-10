#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段2：累積異常報酬（CAR）計算
使用日頻數據和貝氏縮減CAPM模型計算CAR(-3, +5)
符合claude.md日頻事件研究法架構
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 檢測專案根目錄（包含output_abetad1的目錄）
def get_project_root():
    """取得專案根目錄"""
    current = Path.cwd()
    # 如果當前在src目錄，上移一層
    if current.name == 'src':
        return current.parent
    # 否則假設已在根目錄
    return current

PROJECT_ROOT = get_project_root()


class CARCalculator:
    """累積異常報酬計算器（日頻）"""

    def __init__(
        self,
        event_list_path=None,
        tool_beta_1y=None,
        tool_returns=None,
        window_start=-3,
        window_end=5
    ):
        """
        初始化CAR計算器

        Parameters:
        -----------
        event_list_path : str
            事件列表檔案路徑
        tool_beta_1y : str
            TEJ Beta工具（1年期）
        tool_returns : str
            TEJ日報酬率工具
        window_start : int
            事件窗期起始（相對事件日，預設：-3）
        window_end : int
            事件窗期結束（相對事件日，預設：5）
        """
        # 使用PROJECT_ROOT設定預設路徑
        self.event_list_path = event_list_path or str(PROJECT_ROOT / 'data/processed/event_list.csv')
        self.tool_beta_1y = tool_beta_1y or str(PROJECT_ROOT / 'tej_tool_TWN_ABETAD1.py')
        self.tool_returns = tool_returns or str(PROJECT_ROOT / 'tej_tool_TWN_APRCD2_g.py')
        self.events_df = None
        self.car_results = []
        
        # 儲存窗期參數
        self.window_start = window_start
        self.window_end = window_end
        print(f"CAR計算窗期: [{window_start}, {window_end}]")

    def load_events(self):
        """載入事件列表"""
        print(f"載入事件列表: {self.event_list_path}")
        self.events_df = pd.read_csv(self.event_list_path)
        self.events_df['mdate'] = pd.to_datetime(self.events_df['mdate'])
        print(f"總共 {len(self.events_df)} 筆事件\n")
        return self

    def get_beta_and_returns_data(self, coid, event_date):
        """
        從 ABETAD1 取得 Beta 值和報酬率資料（roi, wroi）

        根據claude.md，使用貝氏縮減：β_shrunk = 0.7 × β_1yr + 0.3 × β_3yr

        Parameters:
        -----------
        coid : str
            股票代號
        event_date : datetime
            事件日期

        Returns:
        --------
        tuple : (beta_shrunk, returns_df)
            - beta_shrunk: 縮減後的Beta值
            - returns_df: 包含 roi (個股報酬率) 和 wroi (市場報酬率) 的 DataFrame
        """
        try:
            # 🚀 優化：使用固定365天快取視窗（減少重複API呼叫）
            buffer_days = 365
            start_date = (event_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            end_date = (event_date + timedelta(days=180)).strftime('%Y-%m-%d')

            # 檢查輸出目錄是否已有檔案
            output_dir = PROJECT_ROOT / 'output_abetad1'
            output_dir.mkdir(exist_ok=True)
            existing_files = list(output_dir.glob(f'ABETAD1_{coid}_*.csv'))

            print(f"    檢查目錄: {output_dir}")
            print(f"    找到 {len(existing_files)} 個 ABETAD1 檔案")

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
                        print(f"    使用現有檔案 (涵蓋範圍: {cache_start.date()} 到 {cache_end.date()})")
                except:
                    use_cache = False

            # 如果沒有可用的 cache，就呼叫工具
            if not use_cache:
                print(f"    呼叫 TEJ ABETAD1 API...")
                cmd = [
                    'python3', self.tool_beta_1y,
                    '-c', str(coid),
                    '-start', start_date,
                    '-end', end_date
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)
                existing_files = list(output_dir.glob(f'ABETAD1_{coid}_*.csv'))

            if not existing_files:
                return None, None

            # 讀取 ABETAD1 資料
            latest_file = max(existing_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            df['mdate'] = pd.to_datetime(df['mdate'])

            # 取最新的1年期和3年期Beta（用於貝氏縮減）
            if 'beta_1y' in df.columns:
                beta_1y = df['beta_1y'].iloc[-1]
            else:
                print(f"    Beta_1y 欄位不存在")
                return None, None

            if 'beta_3y' in df.columns:
                beta_3y = df['beta_3y'].iloc[-1]
            else:
                beta_3y = beta_1y
                print(f"    注意：無3年期Beta，使用1年期Beta替代")

            # 處理NaN值
            if pd.isna(beta_1y):
                beta_1y = 1.0
            if pd.isna(beta_3y):
                beta_3y = beta_1y if not pd.isna(beta_1y) else 1.0

            # 貝氏縮減
            beta_shrunk = 0.7 * beta_1y + 0.3 * beta_3y

            # 檢查必要的報酬率欄位
            if 'roi' not in df.columns or 'wroi' not in df.columns:
                print(f"    警告：缺少 roi 或 wroi 欄位")
                return beta_shrunk, None

            return beta_shrunk, df

        except Exception as e:
            print(f"  [Beta+報酬率] 錯誤：無法取得{coid}的 ABETAD1 資料")
            print(f"  [Beta+報酬率] 詳細錯誤: {str(e)}")
            return None, None

    def calculate_car(self, coid, event_date, beta_shrunk, returns_df):
        """
        計算單一事件的CAR（使用動態窗期和 ABETAD1 報酬率）

        Parameters:
        -----------
        coid : str, 股票代號
        event_date : datetime, 事件日期
        beta_shrunk : float, 縮減後的Beta值
        returns_df : DataFrame, 包含 roi 和 wroi 的報酬率資料

        Returns:
        --------
        dict : 包含CAR的結果字典
        - CAR = Σ AR_{i,t} from t=window_start to t=window_end
        - AR_{i,t} = R_{i,t} - β_shrunk × R_{m,t}
        - 使用 ABETAD1 的 roi (個股報酬率) 和 wroi (市場指數報酬率)
        """
        if returns_df is None or len(returns_df) == 0:
            print(f"  [2/2] 報酬率資料為空")
            return None

        # 定位事件日及窗期
        returns_df = returns_df.sort_values('mdate').reset_index(drop=True)

        # 找到事件日的索引位置
        event_mask = returns_df['mdate'] == event_date
        if event_mask.any():
            event_idx = returns_df[event_mask].index[0]
        else:
            # 事件日無交易資料，尋找最近交易日
            returns_df['date_diff'] = abs((returns_df['mdate'] - event_date).dt.days)
            event_idx = returns_df['date_diff'].idxmin()
            returns_df = returns_df.drop('date_diff', axis=1)

        # 取得事件日前後的報酬率（使用動態窗期）
        window_start_idx = max(0, event_idx + self.window_start)
        window_end_idx = min(len(returns_df), event_idx + self.window_end + 1)

        window_df = returns_df.iloc[window_start_idx:window_end_idx].copy()

        if len(window_df) < 5:  # 至少需要5個交易日
            print(f"  [2/2] 交易日數不足: {len(window_df)} < 5")
            return None

        # 檢查必要欄位（使用 ABETAD1 的 roi 和 wroi）
        if 'roi' not in window_df.columns or 'wroi' not in window_df.columns:
            print(f"  [2/2] 缺少報酬率欄位 (roi, wroi)")
            print(f"  [2/2] 可用欄位: {list(window_df.columns)}")
            return None

        # 計算AR：AR = R_i - β × R_m
        # 使用 ABETAD1 的 roi (個股報酬率) 和 wroi (市場指數報酬率，根據上市/上櫃自動選擇)
        window_df['R_i'] = window_df['roi']      # 個股日報酬率（%）
        window_df['R_m'] = window_df['wroi']     # 市場指數日報酬率（%）
        window_df['Expected_Return'] = beta_shrunk * window_df['R_m']  # 預期報酬（%）
        window_df['AR'] = window_df['R_i'] - window_df['Expected_Return']  # 異常報酬（%）

        # 計算CAR（使用動態窗期）
        car = window_df['AR'].sum()

        print(f"  [2/2] 計算完成: {len(window_df)} 個交易日，CAR = {car:.4f}%")

        # 動態生成欄位名稱
        car_column_name = f'CAR_m{abs(self.window_start)}_p{self.window_end}'
        
        return {
            'coid': coid,
            'event_date': event_date,
            'beta_shrunk': beta_shrunk,
            car_column_name: car,
            'n_days': len(window_df)
        }

    def calculate_car_for_event(self, coid, event_date):
        """
        計算單一事件的完整CAR流程（使用 ABETAD1 取得 Beta 和報酬率）

        Parameters:
        -----------
        coid : str
            股票代號
        event_date : datetime
            事件日期

        Returns:
        --------
        dict or None
            包含CAR結果的字典，失敗則返回None
        """
        # 1. 從 ABETAD1 取得 Beta 和報酬率
        print(f"  [1/2] 取得 Beta 和報酬率...")
        beta_shrunk, returns_df = self.get_beta_and_returns_data(coid, event_date)
        if beta_shrunk is None or returns_df is None:
            print(f"  [1/2] Beta 或報酬率取得失敗")
            return None
        print(f"  [1/2] Beta = {beta_shrunk:.4f}, 報酬率筆數 = {len(returns_df)}")

        # 2. 計算CAR
        return self.calculate_car(coid, event_date, beta_shrunk, returns_df)

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
        print(f"開始計算CAR（窗期：{self.window_start} 到 {self.window_end}）")
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
                
                future = executor.submit(self.calculate_car_for_event, coid, event_date)
                future_to_event[future] = (idx, coid, event_date)
            
            # 收集結果（按完成順序）
            completed = 0
            for future in as_completed(future_to_event):
                idx, coid, event_date = future_to_event[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.car_results.append(result)
                        car_col = [k for k in result.keys() if k.startswith('CAR_')][0]
                        print(f"✓ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - {car_col}: {result[car_col]:.4f}%")
                    else:
                        print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 無法計算CAR")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 錯誤: {e}")

        print(f"\n完成！成功計算 {len(self.car_results)} 筆事件的 CAR\n")

    def save_results(self, output_path=None):
        """儲存CAR結果"""
        output_path = output_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.car_results:
            print("警告：沒有CAR結果可儲存")
            return

        df = pd.DataFrame(self.car_results)
        df.to_csv(output_path, index=False)

        print(f"CAR資料已儲存至: {output_path}")
        print(f"共 {len(df)} 筆結果\n")

        # 摘要統計（動態取得CAR欄位）
        car_col = [c for c in df.columns if c.startswith('CAR_')][0]
        print("=== CAR摘要統計 ===\n")
        print(f"平均: {df[car_col].mean():.4f}%")
        print(f"中位數: {df[car_col].median():.4f}%")
        print(f"標準差: {df[car_col].std():.4f}%")
        print(f"最小值: {df[car_col].min():.4f}%")
        print(f"最大值: {df[car_col].max():.4f}%")


def main():
    """主程式"""
    print("="*80)
    print("階段2：CAR計算（日頻，動態窗期）")
    print("="*80)
    print()

    # 測試：使用預設窗期 -3 到 +5
    calculator = CARCalculator(window_start=-3, window_end=5)
    calculator.load_events()
    calculator.process_events(sample_size=10)  # 測試：先處理10筆
    calculator.save_results()

    print("階段2完成！\n")


if __name__ == '__main__':
    main()
