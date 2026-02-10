#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段7：X6、X7、X8計算（控制變數）
X6: 公司規模（市值對數）
X7: 帳面市值比（B/M Ratio）
X8: 產業別（虛擬變數）
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_project_root():
    current = Path.cwd()
    return current.parent if current.name == 'src' else current

PROJECT_ROOT = get_project_root()


class ControlVariablesCalculator:
    def __init__(self, event_list_path=None, car_data_path=None, tool_aifin=None, tool_aifina=None, tool_aind=None):
        self.event_list_path = event_list_path or (PROJECT_ROOT / 'data/processed/event_list.csv')
        self.car_data_path = car_data_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        self.tool_aifin = tool_aifin or str(PROJECT_ROOT / 'tej_tool_TWN_AIFIN.py')
        self.tool_aifina = tool_aifina or str(PROJECT_ROOT / 'tej_tool_TWN_AIFINA.py')
        self.tool_aind = tool_aind or str(PROJECT_ROOT / 'tej_tool_TWN_AIND.py')
        self.events_df = None
        self.industry_df = None  # 產業分類查找表
        self.control_results = []

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

        # 一次性查詢所有公司的產業分類
        self.load_industry_data()
        return self

    def load_industry_data(self):
        """一次性查詢所有公司的TEJ產業分類"""
        try:
            output_dir = PROJECT_ROOT / 'output_aind'
            output_dir.mkdir(exist_ok=True)

            # 檢查是否已有 all companies 檔案 (兩種命名格式)
            all_files = list(output_dir.glob('AIND_all_*.csv')) + list(output_dir.glob('company_basic_all_*.csv'))

            if not all_files:
                print("查詢所有公司的TEJ產業分類...")
                cmd = ['python3', self.tool_aind, '--all', '-o', str(output_dir)]
                subprocess.run(cmd, capture_output=True, timeout=120)
                all_files = list(output_dir.glob('AIND_all_*.csv')) + list(output_dir.glob('company_basic_all_*.csv'))

            if all_files:
                latest_file = max(all_files, key=os.path.getctime)
                self.industry_df = pd.read_csv(latest_file)
                print(f"載入 {len(self.industry_df)} 家公司的產業分類")
            else:
                print("警告：無法取得產業分類資料")

        except Exception as e:
            print(f"產業分類載入失敗: {str(e)}")

    def get_industry_classification(self, coid):
        """從產業分類查找表取得產業別"""
        if self.industry_df is None or len(self.industry_df) == 0:
            return None

        try:
            # 查找該 coid 的產業分類（轉為字串以匹配TEJ資料格式）
            company_data = self.industry_df[self.industry_df['coid'] == str(coid)]

            if len(company_data) > 0:
                # 優先使用 tejind2_c (二級產業)
                if 'tejind2_c' in company_data.columns:
                    ind = company_data['tejind2_c'].iloc[0]
                    if pd.notna(ind):
                        return ind

                # 備選：tejind1_c (一級產業)
                if 'tejind1_c' in company_data.columns:
                    ind = company_data['tejind1_c'].iloc[0]
                    if pd.notna(ind):
                        return ind

            return None

        except Exception as e:
            return None

    def get_financial_data(self, coid, event_date):
        """取得財務資料（市值、淨值、產業別）"""
        try:
            # X6: 從APRCD1取得市值（mv欄位，單位：百萬元）
            output_dir_aprcd1 = PROJECT_ROOT / 'output_aprcd1'
            output_dir_aprcd1.mkdir(exist_ok=True)

            # Check for existing APRCD1 files
            aprcd1_files = list(output_dir_aprcd1.glob(f'APRCD1_{coid}_*.csv'))
            if not aprcd1_files:
                aprcd1_files = list(output_dir_aprcd1.glob(f'aprcd1_{coid}_*.csv'))
            if not aprcd1_files:
                aprcd1_files = list(PROJECT_ROOT.glob(f'APRCD1_{coid}_*.csv'))

            x6_log_size = None
            x7_bm_ratio = None
            x8_industry = None

            if aprcd1_files:
                df_price = pd.read_csv(max(aprcd1_files, key=os.path.getctime))
                df_price['mdate'] = pd.to_datetime(df_price['mdate'])

                # Get T-1 data (one day before event)
                t_minus_1 = event_date - timedelta(days=1)
                price_data = df_price[df_price['mdate'] <= t_minus_1].tail(1)

                if len(price_data) > 0:
                    # X6: Log of market cap
                    if 'mv' in price_data.columns:
                        market_cap = price_data['mv'].iloc[0]  # 單位：百萬元
                        if market_cap and market_cap > 0:
                            x6_log_size = np.log(market_cap)

                    # X7: B/M ratio = 1 / P/B ratio
                    # APRCD1 has pbr_tej (Price-to-Book Ratio from TEJ)
                    if 'pbr_tej' in price_data.columns:
                        pbr = price_data['pbr_tej'].iloc[0]
                        if pbr and pbr > 0:
                            x7_bm_ratio = 1.0 / pbr  # B/M = 1/P/B
                    elif 'pbr_tse' in price_data.columns:
                        pbr = price_data['pbr_tse'].iloc[0]
                        if pbr and pbr > 0:
                            x7_bm_ratio = 1.0 / pbr

            # X8: 從產業分類查找表取得產業別
            x8_industry = self.get_industry_classification(coid)

            return x6_log_size, x7_bm_ratio, x8_industry

        except Exception as e:
            print(f"  [控制變數] 錯誤: {str(e)}")
            return None, None, None

    def calculate_controls_for_event(self, coid, event_date):
        """計算單一事件的X6、X7、X8"""
        x6_log_size, x7_bm_ratio, x8_industry_name = self.get_financial_data(coid, event_date)

        return {
            'coid': coid,
            'event_date': event_date,
            'X6_log_size': x6_log_size,
            'X7_bm_ratio': x7_bm_ratio,
            'X8_industry_name': x8_industry_name,
            'X8_industry': x8_industry_name  # Keep both for compatibility
        }

    def process_events(self, sample_size=None, start_date='2020-01-01', end_date='2025-12-31'):
        print("="*80)
        print("階段7：X6、X7、X8計算（控制變數）")
        print("="*80)

        # 使用參數化的日期範圍過濾
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        events_filtered = self.events_df[
            (self.events_df['mdate'] >= start_ts) &
            (self.events_df['mdate'] <= end_ts)
        ].copy()
        # 統一排序：先按日期、再按股票代號
        events_sorted = events_filtered.sort_values(['mdate', 'coid'], ascending=True)
        events_to_process = events_sorted.head(sample_size) if sample_size else events_sorted

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
                
                future = executor.submit(self.calculate_controls_for_event, coid, event_date)
                future_to_event[future] = (idx, coid, event_date)
            
            # 收集結果（按完成順序）
            completed = 0
            for future in as_completed(future_to_event):
                idx, coid, event_date = future_to_event[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.control_results.append(result)
                        print(f"✓ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')}")
                    else:
                        print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 無法計算X7/X8/X9")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 錯誤: {e}")

        print(f"\n完成！成功計算 {len(self.control_results)} 筆\n")

    def save_results(self, output_path=None):
        output_path = output_path or (PROJECT_ROOT / 'data/processed/x6_x7_x8_controls.csv')
        if not self.control_results:
            print("無控制變數結果")
            return
        df = pd.DataFrame(self.control_results)
        df.to_csv(output_path, index=False)
        print(f"X6/X7/X8已儲存: {output_path}")


def main():
    calculator = ControlVariablesCalculator()
    calculator.load_events()
    calculator.process_events(sample_size=35)
    calculator.save_results()


if __name__ == '__main__':
    main()
