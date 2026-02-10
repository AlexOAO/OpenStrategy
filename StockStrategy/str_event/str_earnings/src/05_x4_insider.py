#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段5：X4計算（十大股東持股變化率）與散戶持股分布
以ABSTN1取得十大股東（不含董監）持股變化率，並以ADCSHR計算散戶集中度
"""

import pandas as pd
from datetime import timedelta
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_project_root():
    current = Path.cwd()
    return current.parent if current.name == 'src' else current

PROJECT_ROOT = get_project_root()


class InsiderTradingCalculator:
    def __init__(
        self,
        event_list_path: Optional[Path] = None,
        car_data_path: Optional[Path] = None,
        tool_abstn1: Optional[str] = None,
        tool_adcshr: Optional[str] = None,
        lookback_days: int = 365,  # 🚀 優化：增加快取視窗至365天
    ):
        self.event_list_path = event_list_path or (PROJECT_ROOT / 'data/processed/event_list.csv')
        self.car_data_path = car_data_path or (PROJECT_ROOT / 'data/processed/car_data.csv')
        self.tool_abstn1 = tool_abstn1 or str(PROJECT_ROOT / 'tej_tool_TWN_ABSTN1.py')
        self.tool_adcshr = tool_adcshr or str(PROJECT_ROOT / 'tej_tool_TWN_ADCSHR.py')
        self.lookback_days = lookback_days
        self.events_df = None
        self.x4_results = []
        self.abstn1_output_dir = PROJECT_ROOT / 'output_abstn1'
        self.adcshr_output_dir = PROJECT_ROOT / 'output_adcshr'

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

    def _ensure_abstn1_file(self, coid: str, event_date: pd.Timestamp) -> Optional[Path]:
        self.abstn1_output_dir.mkdir(exist_ok=True)
        start_date = (event_date - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        end_date = (event_date - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 檢查多種可能的檔名（basic 或 all 群組）
        possible_files = [
            self.abstn1_output_dir / f"abstn1_{coid}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_all.csv",
            self.abstn1_output_dir / f"abstn1_{coid}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_basic.csv",
            self.abstn1_output_dir / f"abstn1_{coid}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_management.csv",
        ]
        
        # 檢查是否已有可用檔案
        existing_file = None
        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # 檢查是否有數據且日期範圍足夠
                    if len(df) > 0 and 'mdate' in df.columns:
                        df['mdate'] = pd.to_datetime(df['mdate'])
                        cache_start = df['mdate'].min()
                        cache_end = df['mdate'].max()
                        required_start = pd.Timestamp(start_date)
                        required_end = pd.Timestamp(end_date)
                        # cache 範圍足夠且包含必要欄位
                        if cache_start <= required_start and cache_end >= required_end:
                            # 檢查是否有 fld008 欄位（十大股東）
                            if 'fld008' in df.columns:
                                existing_file = file_path
                                break
                except:
                    continue
        
        if existing_file:
            return existing_file
        
        # 需要重新抓取，使用 all 群組確保包含所有欄位
        file_path = possible_files[0]  # 使用 all 群組的檔名
        cmd = [
            'python3',
            self.tool_abstn1,
            '-s', str(coid),
            '--start-date', start_date,
            '--end-date', end_date,
            '-f', 'all',
        ]
        try:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=120)
            if result.returncode != 0:
                print(f"  [ABSTN1] 下載失敗 {coid}: {result.stderr.decode(errors='ignore').strip()}")
        except Exception as exc:
            print(f"  [ABSTN1] 呼叫失敗 {coid}: {exc}")

        return file_path if file_path.exists() else None

    def _ensure_adcshr_file(self, coid: str, event_date: pd.Timestamp) -> Optional[Path]:
        self.adcshr_output_dir.mkdir(exist_ok=True)
        start_date = (event_date - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        end_date = (event_date - timedelta(days=1)).strftime('%Y-%m-%d')
        filename = f"ADCSHR_{coid}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        file_path = self.adcshr_output_dir / filename

        # 檢查檔案是否存在且有效
        need_fetch = True
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # 檢查是否有數據且日期範圍足夠
                if len(df) > 0 and 'mdate' in df.columns:
                    df['mdate'] = pd.to_datetime(df['mdate'])
                    cache_start = df['mdate'].min()
                    cache_end = df['mdate'].max()
                    required_start = pd.Timestamp(start_date)
                    required_end = pd.Timestamp(end_date)
                    # cache 範圍足夠就不用重抓
                    if cache_start <= required_start and cache_end >= required_end:
                        need_fetch = False
            except:
                need_fetch = True

        if need_fetch:
            cmd = [
                'python3',
                self.tool_adcshr,
                '--coid', str(coid),
                '--start-date', start_date,
                '--end-date', end_date,
            ]
            try:
                result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=120)
                if result.returncode != 0:
                    print(f"  [ADCSHR] 下載失敗 {coid}: {result.stderr.decode(errors='ignore').strip()}")
            except Exception as exc:
                print(f"  [ADCSHR] 呼叫失敗 {coid}: {exc}")

        return file_path if file_path.exists() else None

    @staticmethod
    def _calc_top10_change(df: pd.DataFrame, event_date: pd.Timestamp) -> Optional[float]:
        """
        計算十大股東（不含董監）持股變化率（X4）
        變化率 = (期末持股 - 期初持股) / 期初持股
        """
        df = df.copy()
        df['mdate'] = pd.to_datetime(df['mdate'])
        t_minus_60 = event_date - timedelta(days=60)
        t_minus_1 = event_date - timedelta(days=1)

        df_t60 = df[df['mdate'] <= t_minus_60].tail(1)
        df_t1 = df[df['mdate'] <= t_minus_1].tail(1)

        if df_t60.empty or df_t1.empty:
            return None

        # fld008: 十大股東持股(不含董監) 股數
        holding_field = 'fld008' if 'fld008' in df.columns else None
        if holding_field is None:
            return None

        holding_t60 = df_t60[holding_field].iloc[0]
        holding_t1 = df_t1[holding_field].iloc[0]
        
        # 計算變化率：(期末 - 期初) / 期初
        if pd.isna(holding_t60) or pd.isna(holding_t1) or holding_t60 == 0:
            return None
        
        change_rate = (holding_t1 - holding_t60) / holding_t60 * 100.0  # 轉為百分比
        return change_rate

    @staticmethod
    def _calc_retail_ratio(df: pd.DataFrame, event_date: pd.Timestamp) -> Optional[float]:
        """
        計算散戶持股比例變化率（X6變數）

        散戶定義：400張以下
        變化率 = (當期散戶持股% - 上期散戶持股%) / 上期散戶持股%
        
        包含級距：1張以下(a), 1-5張(b), 5-10張(c), 10-15張(d), 15-20張(e),
                 20-30張(f), 30-40張(ga), 40-50張(gb), 50-100張(h),
                 100-200張(i), 200-400張(j)
        """
        df = df.copy()
        df['mdate'] = pd.to_datetime(df['mdate'])
        
        # T-1 (當期) 和 T-60 (上期)
        t_minus_1 = event_date - timedelta(days=1)
        t_minus_60 = event_date - timedelta(days=60)
        
        current = df[df['mdate'] <= t_minus_1].tail(1)
        previous = df[df['mdate'] <= t_minus_60].tail(1)
        
        if current.empty or previous.empty:
            return None

        # 400張以下的所有級距（持股比例欄位以03結尾）
        required_cols = {'a03', 'b03', 'c03', 'd03', 'e03', 'f03',
                        'ga03', 'gb03', 'h03', 'i03', 'j03'}

        if not required_cols.issubset(current.columns) or not required_cols.issubset(previous.columns):
            return None

        # 加總400張以下的持股比例
        retail_ratio_current = float(current[list(required_cols)].iloc[0].sum())
        retail_ratio_previous = float(previous[list(required_cols)].iloc[0].sum())
        
        # 計算變化率：(當期 - 上期) / 上期
        if pd.isna(retail_ratio_previous) or pd.isna(retail_ratio_current) or retail_ratio_previous == 0:
            return None
        
        change_rate = (retail_ratio_current - retail_ratio_previous) / retail_ratio_previous * 100.0
        return change_rate

    def calculate_x4_for_event(self, coid, event_date):
        try:
            abstn1_file = self._ensure_abstn1_file(str(coid), event_date)
            adcshr_file = self._ensure_adcshr_file(str(coid), event_date)

            top10_change = None
            retail_ratio = None

            if abstn1_file is not None:
                df_abstn1 = pd.read_csv(abstn1_file)
                top10_change = self._calc_top10_change(df_abstn1, event_date)

            if adcshr_file is not None:
                df_adcshr = pd.read_csv(adcshr_file)
                retail_ratio = self._calc_retail_ratio(df_adcshr, event_date)

            # 即使數據部分缺失也保留記錄（用 NaN 表示）
            # 這樣可以在後續分析中處理缺失值
            return {
                'coid': coid,
                'event_date': event_date,
                'X4_top10_change_rate': top10_change,  # 十大股東持股變化率
                'X6_retail_change_rate': retail_ratio,  # 散戶持股比例變化率（原X9改為X6）
            }

        except Exception as e:
            print(f"  [階段5] 錯誤 ({coid}): {str(e)}")
            return None

    def process_events(self, sample_size=None, start_date='2020-01-01', end_date='2025-12-31'):
        print("="*80)
        print("階段5：X4（十大股東持股變化率）與X6（散戶持股變化率）計算")
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
        events_to_process = events_sorted.head(sample_size) if sample_size else events_sorted

        # 統計變數
        x4_success = 0  # X4 成功取得數量
        x6_success = 0  # X6 成功取得數量
        all_success = 0  # 兩者都成功的數量
        
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
                
                future = executor.submit(self.calculate_x4_for_event, coid, event_date)
                future_to_event[future] = (idx, coid, event_date)
            
            # 收集結果（按完成順序）
            completed = 0
            for future in as_completed(future_to_event):
                idx, coid, event_date = future_to_event[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.x4_results.append(result)
                        
                        # 更新統計
                        has_x4 = result['X4_top10_change_rate'] is not None and \
                                 not pd.isna(result['X4_top10_change_rate'])
                        has_x6 = result['X6_retail_change_rate'] is not None and \
                                 not pd.isna(result['X6_retail_change_rate'])
                        
                        if has_x4:
                            x4_success += 1
                        if has_x6:
                            x6_success += 1
                        if has_x4 and has_x6:
                            all_success += 1
                        
                        # 狀態顯示
                        status = []
                        if has_x4:
                            status.append("X4✓")
                        if has_x6:
                            status.append("X6✓")
                        
                        status_str = ', '.join(status) if status else '無數據'
                        print(f"✓ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} [{status_str}]")
                    else:
                        print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} [處理失敗]")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] {coid} @ {event_date.strftime('%Y-%m-%d')} - 錯誤: {e}")

        # 最終統計
        print(f"\n{'='*80}")
        print(f"階段5完成統計:")
        print(f"  總處理事件: {total}")
        print(f"  成功記錄: {len(self.x4_results)}")
        print(f"  X4 (十大股東變化率) 成功率: {x4_success}/{total} ({x4_success/total*100:.1f}%)")
        print(f"  X6 (散戶變化率) 成功率: {x6_success}/{total} ({x6_success/total*100:.1f}%)")
        print(f"  兩者皆有: {all_success}/{total} ({all_success/total*100:.1f}%)")
        print(f"{'='*80}\n")

    def save_results(self, output_path=None):
        output_path = output_path or (PROJECT_ROOT / 'data/processed/x4_insider.csv')
        if not self.x4_results:
            print("無X4結果")
            return
        df = pd.DataFrame(self.x4_results)
        df.to_csv(output_path, index=False)
        print(f"X4/X6已儲存: {output_path}")


def main():
    calculator = InsiderTradingCalculator()
    calculator.load_events()
    calculator.process_events(sample_size=35)
    calculator.save_results()


if __name__ == '__main__':
    main()
