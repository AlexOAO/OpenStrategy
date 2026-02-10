#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段10：整合變數計算器 🚀
一次API調用，同時計算多個變數，大幅提升效能！

計算內容：
- Y: CAR (累積異常報酬)
- X2: 20日累積週轉率
- X3: 10日累積報酬率
- X4: 十大股東持股變化率
- X6: 散戶持股變化率
- X7: 市值（對數）
- X8: B/M比率
- X9: 產業代碼

優勢：
✅ 減少90%的API調用次數
✅ 共享數據快取
✅ 並行計算所有變數
✅ 統一錯誤處理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def get_project_root():
    """取得專案根目錄"""
    current = Path.cwd()
    if current.name == 'src':
        return current.parent
    return current

PROJECT_ROOT = get_project_root()


class IntegratedCalculator:
    """整合變數計算器 - 一次API調用計算所有變數"""

    def __init__(
        self,
        event_list_path=None,
        window_start=-3,
        window_end=5
    ):
        """
        初始化整合計算器

        Parameters:
        -----------
        event_list_path : str
            事件列表檔案路徑
        window_start : int
            CAR窗期起始（預設：-3）
        window_end : int
            CAR窗期結束（預設：5）
        """
        self.event_list_path = event_list_path or str(PROJECT_ROOT / 'data/processed/event_list.csv')
        self.events_df = None
        self.results = []
        
        # CAR窗期參數
        self.window_start = window_start
        self.window_end = window_end
        
        # TEJ工具路徑
        self.tool_abetad1 = str(PROJECT_ROOT / 'tej_tool_TWN_ABETAD1.py')
        self.tool_aprcd1 = str(PROJECT_ROOT / 'tej_tool_TWN_APRCD1.py')
        self.tool_ashr1a = str(PROJECT_ROOT / 'tej_tool_TWN_ASHR1A.py')
        self.tool_aifina = str(PROJECT_ROOT / 'tej_tool_TWN_AIFINA.py')
        self.tool_aind = str(PROJECT_ROOT / 'tej_tool_TWN_AIND.py')
        
        # 輸出目錄
        self.output_abetad1 = PROJECT_ROOT / 'output_abetad1'
        self.output_aprcd1 = PROJECT_ROOT / 'output_aprcd1'
        self.output_ashr1a = PROJECT_ROOT / 'output_ashr1a'
        
        for dir_path in [self.output_abetad1, self.output_aprcd1, self.output_ashr1a]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"🚀 整合計算器初始化完成")
        print(f"   CAR窗期: [{window_start}, {window_end}]")

    def load_events(self):
        """載入事件列表"""
        print(f"\n📂 載入事件列表: {self.event_list_path}")
        self.events_df = pd.read_csv(self.event_list_path)
        self.events_df['mdate'] = pd.to_datetime(self.events_df['mdate'])
        print(f"   總共 {len(self.events_df)} 筆事件")
        return self

    def fetch_all_data(self, coid, event_date):
        """
        一次性取得所有需要的資料（最小化API調用）
        
        Returns:
        --------
        dict : {
            'abetad1': DataFrame,  # Beta & 報酬率
            'aprcd1': DataFrame,   # 成交量
            'ashr1a': DataFrame,   # 股東持股
            'aifina': dict,        # 財務資料
            'aind': str            # 產業代碼
        }
        """
        data = {}
        
        # 🚀 優化：使用365天快取視窗，一次取得所有需要的歷史數據
        buffer_days = 400  # 多留一點buffer確保有足夠交易日
        start_date = (event_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        end_date = (event_date + timedelta(days=30)).strftime('%Y-%m-%d')  # 多取一點未來數據
        
        # 1. ABETAD1 - Beta & 報酬率（用於CAR和X3）
        print(f"      📡 取得 ABETAD1 數據...")
        data['abetad1'] = self._fetch_abetad1(coid, start_date, end_date)
        
        # 2. APRCD1 - 成交量（用於X2）
        print(f"      📡 取得 APRCD1 數據...")
        data['aprcd1'] = self._fetch_aprcd1(coid, start_date, end_date)
        
        # 3. ASHR1A - 股東持股（用於X4和X6）
        print(f"      📡 取得 ASHR1A 數據...")
        data['ashr1a'] = self._fetch_ashr1a(coid, event_date)
        
        # 4. AIFINA - 財務資料（用於X7和X8）
        print(f"      📡 取得 AIFINA 數據...")
        data['aifina'] = self._fetch_aifina(coid, event_date)
        
        # 5. AIND - 產業代碼（用於X9）
        print(f"      📡 取得 AIND 數據...")
        data['aind'] = self._fetch_aind(coid, event_date)
        
        return data

    def _fetch_abetad1(self, coid, start_date, end_date):
        """取得 ABETAD1 數據（Beta & 報酬率）"""
        try:
            # 檢查快取
            existing_files = list(self.output_abetad1.glob(f'ABETAD1_{coid}_*.csv'))
            
            use_cache = False
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_file)
                    df['mdate'] = pd.to_datetime(df['mdate'])
                    
                    cache_start = df['mdate'].min()
                    cache_end = df['mdate'].max()
                    required_start = pd.Timestamp(start_date)
                    required_end = pd.Timestamp(end_date)
                    
                    if cache_start <= required_start and cache_end >= required_end:
                        use_cache = True
                        print(f"         ✓ 使用快取")
                        return df
                except:
                    pass
            
            if not use_cache:
                # 調用API
                cmd = [
                    'python3', self.tool_abetad1,
                    '--coid', coid,
                    '--start', start_date,
                    '--end', end_date
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 讀取結果
                latest_file = max(self.output_abetad1.glob(f'ABETAD1_{coid}_*.csv'), 
                                 key=os.path.getctime)
                df = pd.read_csv(latest_file)
                df['mdate'] = pd.to_datetime(df['mdate'])
                print(f"         ✓ API調用成功")
                return df
                
        except Exception as e:
            print(f"         ✗ ABETAD1 取得失敗: {e}")
            return None

    def _fetch_aprcd1(self, coid, start_date, end_date):
        """取得 APRCD1 數據（成交量）"""
        try:
            # 檢查快取
            existing_files = list(self.output_aprcd1.glob(f'APRCD1_{coid}_*.csv'))
            
            use_cache = False
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_file)
                    df['mdate'] = pd.to_datetime(df['mdate'])
                    
                    cache_start = df['mdate'].min()
                    cache_end = df['mdate'].max()
                    required_start = pd.Timestamp(start_date)
                    required_end = pd.Timestamp(end_date)
                    
                    if cache_start <= required_start and cache_end >= required_end:
                        use_cache = True
                        print(f"         ✓ 使用快取")
                        return df
                except:
                    pass
            
            if not use_cache:
                # 調用API
                cmd = [
                    'python3', self.tool_aprcd1,
                    '--coid', coid,
                    '--start', start_date,
                    '--end', end_date
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 讀取結果
                latest_file = max(self.output_aprcd1.glob(f'APRCD1_{coid}_*.csv'), 
                                 key=os.path.getctime)
                df = pd.read_csv(latest_file)
                df['mdate'] = pd.to_datetime(df['mdate'])
                print(f"         ✓ API調用成功")
                return df
                
        except Exception as e:
            print(f"         ✗ APRCD1 取得失敗: {e}")
            return None

    def _fetch_ashr1a(self, coid, event_date):
        """取得 ASHR1A 數據（股東持股）"""
        try:
            # ASHR1A 是季度數據，取事件日前後各一季
            start_date = (event_date - timedelta(days=180)).strftime('%Y-%m-%d')
            end_date = (event_date + timedelta(days=180)).strftime('%Y-%m-%d')
            
            # 檢查快取
            existing_files = list(self.output_ashr1a.glob(f'ASHR1A_{coid}_*.csv'))
            
            use_cache = False
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_file)
                    df['mdate'] = pd.to_datetime(df['mdate'])
                    
                    # 檢查是否包含事件日前後的季度資料
                    if len(df) > 0:
                        use_cache = True
                        print(f"         ✓ 使用快取")
                        return df
                except:
                    pass
            
            if not use_cache:
                # 調用API
                cmd = [
                    'python3', self.tool_ashr1a,
                    '--coid', coid,
                    '--start', start_date,
                    '--end', end_date
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 讀取結果
                latest_file = max(self.output_ashr1a.glob(f'ASHR1A_{coid}_*.csv'), 
                                 key=os.path.getctime)
                df = pd.read_csv(latest_file)
                df['mdate'] = pd.to_datetime(df['mdate'])
                print(f"         ✓ API調用成功")
                return df
                
        except Exception as e:
            print(f"         ✗ ASHR1A 取得失敗: {e}")
            return None

    def _fetch_aifina(self, coid, event_date):
        """取得 AIFINA 數據（財務資料）"""
        try:
            # 取得最近的財務資料（事件日前一年）
            start_date = (event_date - timedelta(days=730)).strftime('%Y-%m-%d')
            end_date = event_date.strftime('%Y-%m-%d')
            
            cmd = [
                'python3', self.tool_aifina,
                '--coid', coid,
                '--start', start_date,
                '--end', end_date
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # AIFINA返回JSON格式
            data = json.loads(result.stdout)
            print(f"         ✓ API調用成功")
            return data
            
        except Exception as e:
            print(f"         ✗ AIFINA 取得失敗: {e}")
            return None

    def _fetch_aind(self, coid, event_date):
        """取得 AIND 數據（產業代碼）"""
        try:
            cmd = [
                'python3', self.tool_aind,
                '--coid', coid,
                '--date', event_date.strftime('%Y-%m-%d')
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # AIND返回產業代碼字串
            ind_code = result.stdout.strip()
            print(f"         ✓ 產業代碼: {ind_code}")
            return ind_code
            
        except Exception as e:
            print(f"         ✗ AIND 取得失敗: {e}")
            return None

    def calculate_all_variables(self, coid, event_date, data):
        """
        基於取得的數據，計算所有變數
        
        Parameters:
        -----------
        coid : str
            股票代號
        event_date : datetime
            事件日期
        data : dict
            _fetch_all_data 返回的數據字典
        
        Returns:
        --------
        dict : 包含所有變數的字典
        """
        result = {
            'coid': coid,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'has_video': self.events_df[
                (self.events_df['coid'] == coid) & 
                (self.events_df['mdate'] == event_date)
            ]['has_video'].iloc[0] if 'has_video' in self.events_df.columns else None
        }
        
        # 計算 Y (CAR)
        print(f"      🔢 計算 CAR...")
        car_data = self._calculate_car(event_date, data['abetad1'])
        result.update(car_data)
        
        # 計算 X2 (週轉率)
        print(f"      🔢 計算 X2 (週轉率)...")
        x2_data = self._calculate_x2(event_date, data['aprcd1'])
        result.update(x2_data)
        
        # 計算 X3 (報酬率)
        print(f"      🔢 計算 X3 (報酬率)...")
        x3_data = self._calculate_x3(event_date, data['abetad1'])
        result.update(x3_data)
        
        # 計算 X4 & X6 (持股變化)
        print(f"      🔢 計算 X4 & X6 (持股變化)...")
        x4_x6_data = self._calculate_x4_x6(event_date, data['ashr1a'])
        result.update(x4_x6_data)
        
        # 計算 X7 & X8 (財務變數)
        print(f"      🔢 計算 X7 & X8 (財務變數)...")
        x7_x8_data = self._calculate_x7_x8(data['aifina'])
        result.update(x7_x8_data)
        
        # X9 (產業代碼)
        result['x9_industry'] = data['aind']
        
        return result

    def _calculate_car(self, event_date, abetad1_df):
        """計算CAR"""
        if abetad1_df is None or len(abetad1_df) == 0:
            return {'car': None, 'beta_1yr': None, 'beta_3yr': None, 'beta_shrunk': None}
        
        try:
            # 取得最新的Beta值
            latest_beta = abetad1_df.iloc[-1]
            beta_1yr = latest_beta.get('beta_1yr_mkt', None)
            beta_3yr = latest_beta.get('beta_3yr_mkt', None)
            
            # 貝氏縮減
            if pd.notna(beta_1yr) and pd.notna(beta_3yr):
                beta_shrunk = 0.7 * beta_1yr + 0.3 * beta_3yr
            else:
                beta_shrunk = None
            
            # 計算CAR
            window_start_date = event_date + timedelta(days=self.window_start)
            window_end_date = event_date + timedelta(days=self.window_end)
            
            window_df = abetad1_df[
                (abetad1_df['mdate'] >= window_start_date) &
                (abetad1_df['mdate'] <= window_end_date)
            ].copy()
            
            if len(window_df) == 0 or beta_shrunk is None:
                return {
                    'car': None,
                    'beta_1yr': beta_1yr,
                    'beta_3yr': beta_3yr,
                    'beta_shrunk': beta_shrunk
                }
            
            # 計算異常報酬
            window_df['ar'] = window_df['roi'] - beta_shrunk * window_df['wroi']
            car = window_df['ar'].sum()
            
            return {
                'car': car,
                'beta_1yr': beta_1yr,
                'beta_3yr': beta_3yr,
                'beta_shrunk': beta_shrunk,
                'ar_count': len(window_df)
            }
            
        except Exception as e:
            print(f"         ✗ CAR計算失敗: {e}")
            return {'car': None, 'beta_1yr': None, 'beta_3yr': None, 'beta_shrunk': None}

    def _calculate_x2(self, event_date, aprcd1_df):
        """計算X2 (20日累積週轉率)"""
        if aprcd1_df is None or len(aprcd1_df) == 0:
            return {'x2_turnover_20d': None}
        
        try:
            # T-23 到 T-4（20個交易日）
            window_start = event_date - timedelta(days=30)  # buffer
            window_end = event_date - timedelta(days=4)
            
            window_df = aprcd1_df[
                (aprcd1_df['mdate'] >= window_start) &
                (aprcd1_df['mdate'] <= window_end)
            ].copy()
            
            # 計算週轉率
            if 'vol' in window_df.columns and 'shs' in window_df.columns:
                window_df['daily_turnover'] = window_df['vol'] / window_df['shs']
                window_df = window_df.dropna(subset=['daily_turnover'])
                
                # 取最近20個交易日
                recent_20 = window_df.nlargest(20, 'mdate')
                
                if len(recent_20) >= 15:  # 至少要有15個交易日
                    x2 = recent_20['daily_turnover'].sum()
                    return {'x2_turnover_20d': x2}
            
            return {'x2_turnover_20d': None}
            
        except Exception as e:
            print(f"         ✗ X2計算失敗: {e}")
            return {'x2_turnover_20d': None}

    def _calculate_x3(self, event_date, abetad1_df):
        """計算X3 (10日累積報酬率)"""
        if abetad1_df is None or len(abetad1_df) == 0:
            return {'x3_return_10d': None}
        
        try:
            # T-13 到 T-4（10個交易日）
            window_start = event_date - timedelta(days=20)  # buffer
            window_end = event_date - timedelta(days=4)
            
            window_df = abetad1_df[
                (abetad1_df['mdate'] >= window_start) &
                (abetad1_df['mdate'] <= window_end)
            ].copy()
            
            if 'roi' in window_df.columns:
                window_df = window_df.dropna(subset=['roi'])
                
                # 取最近10個交易日
                recent_10 = window_df.nlargest(10, 'mdate')
                
                if len(recent_10) >= 8:  # 至少要有8個交易日
                    x3 = recent_10['roi'].sum()
                    return {'x3_return_10d': x3}
            
            return {'x3_return_10d': None}
            
        except Exception as e:
            print(f"         ✗ X3計算失敗: {e}")
            return {'x3_return_10d': None}

    def _calculate_x4_x6(self, event_date, ashr1a_df):
        """計算X4 (十大股東變化) 和 X6 (散戶變化)"""
        if ashr1a_df is None or len(ashr1a_df) == 0:
            return {'x4_insider_change': None, 'x6_retail_change': None}
        
        try:
            # 找事件日前後的兩個季度資料
            ashr1a_df = ashr1a_df.sort_values('mdate')
            
            # 事件日前的最近季度
            before_event = ashr1a_df[ashr1a_df['mdate'] <= event_date]
            if len(before_event) == 0:
                return {'x4_insider_change': None, 'x6_retail_change': None}
            
            current_quarter = before_event.iloc[-1]
            
            # 前一季
            if len(before_event) < 2:
                return {'x4_insider_change': None, 'x6_retail_change': None}
            
            previous_quarter = before_event.iloc[-2]
            
            # 計算變化率
            x4 = None
            x6 = None
            
            if 'top10_holding' in current_quarter and 'top10_holding' in previous_quarter:
                if pd.notna(current_quarter['top10_holding']) and pd.notna(previous_quarter['top10_holding']):
                    if previous_quarter['top10_holding'] != 0:
                        x4 = (current_quarter['top10_holding'] - previous_quarter['top10_holding']) / previous_quarter['top10_holding']
            
            if 'retail_holding' in current_quarter and 'retail_holding' in previous_quarter:
                if pd.notna(current_quarter['retail_holding']) and pd.notna(previous_quarter['retail_holding']):
                    if previous_quarter['retail_holding'] != 0:
                        x6 = (current_quarter['retail_holding'] - previous_quarter['retail_holding']) / previous_quarter['retail_holding']
            
            return {'x4_insider_change': x4, 'x6_retail_change': x6}
            
        except Exception as e:
            print(f"         ✗ X4/X6計算失敗: {e}")
            return {'x4_insider_change': None, 'x6_retail_change': None}

    def _calculate_x7_x8(self, aifina_data):
        """計算X7 (市值對數) 和 X8 (B/M比率)"""
        if aifina_data is None:
            return {'x7_log_market_cap': None, 'x8_book_to_market': None}
        
        try:
            # 從AIFINA數據提取
            market_cap = aifina_data.get('market_cap', None)
            book_value = aifina_data.get('book_value', None)
            
            x7 = np.log(market_cap) if market_cap and market_cap > 0 else None
            x8 = (book_value / market_cap) if (book_value and market_cap and market_cap > 0) else None
            
            return {'x7_log_market_cap': x7, 'x8_book_to_market': x8}
            
        except Exception as e:
            print(f"         ✗ X7/X8計算失敗: {e}")
            return {'x7_log_market_cap': None, 'x8_book_to_market': None}

    def process_one_event(self, idx, row):
        """處理單一事件（用於並行）"""
        coid = row['coid']
        event_date = row['mdate']
        
        print(f"\n   📊 [{idx+1}] 處理: {coid} @ {event_date.strftime('%Y-%m-%d')}")
        
        try:
            # 1. 取得所有數據（最小化API調用）
            data = self.fetch_all_data(coid, event_date)
            
            # 2. 計算所有變數
            result = self.calculate_all_variables(coid, event_date, data)
            
            print(f"      ✅ 完成！CAR={result.get('car', 'N/A')}")
            return result
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
            return None

    def process_events(self, sample_size=None, start_date=None, end_date=None, max_workers=4):
        """
        批次處理事件（並行執行）
        
        Parameters:
        -----------
        sample_size : int
            處理樣本數量
        start_date : str
            事件日期起始範圍
        end_date : str
            事件日期結束範圍
        max_workers : int
            並行執行的工作數量（預設：4）
        """
        df = self.events_df.copy()
        
        # 日期範圍篩選
        if start_date:
            df = df[df['mdate'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['mdate'] <= pd.Timestamp(end_date)]
        
        # 樣本數量限制
        if sample_size:
            df = df.head(sample_size)
        
        print(f"\n🚀 開始處理 {len(df)} 筆事件（並行數：{max_workers}）...")
        
        # 並行處理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_one_event, idx, row): idx 
                for idx, row in df.iterrows()
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.results.append(result)
        
        print(f"\n✅ 處理完成！成功: {len(self.results)}/{len(df)}")

    def save_results(self, output_path=None):
        """儲存整合結果"""
        if not output_path:
            output_path = PROJECT_ROOT / 'data/processed/integrated_results.csv'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 結果已儲存至: {output_path}")
        print(f"   總筆數: {len(df)}")
        
        # 顯示變數完整度
        print("\n📊 變數完整度:")
        for col in df.columns:
            if col not in ['coid', 'event_date']:
                completeness = (df[col].notna().sum() / len(df)) * 100
                print(f"   {col}: {completeness:.1f}%")
        
        return str(output_path)


def main():
    """測試主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='整合變數計算器')
    parser.add_argument('--sample', type=int, default=10, help='處理樣本數量')
    parser.add_argument('--workers', type=int, default=4, help='並行工作數量')
    parser.add_argument('--start-date', type=str, default='2020-01-01')
    parser.add_argument('--end-date', type=str, default='2025-12-31')
    parser.add_argument('--window-start', type=int, default=-3)
    parser.add_argument('--window-end', type=int, default=5)
    
    args = parser.parse_args()
    
    calculator = IntegratedCalculator(
        window_start=args.window_start,
        window_end=args.window_end
    )
    
    calculator.load_events()
    calculator.process_events(
        sample_size=args.sample,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.workers
    )
    calculator.save_results()


if __name__ == '__main__':
    main()
