#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段1：事件識別與分類
根據法說會事件列表建立X1虛擬變數（是否提供公開錄影檔）
符合claude.md日頻事件研究法架構
"""

import pandas as pd
import os


class EventClassifier:
    """法說會事件分類器 - 建立X1變數"""

    def __init__(self, data_path='data/raw/acomtn_all_20200101_20250930.csv'):
        """
        初始化分類器

        Parameters:
        -----------
        data_path : str
            事件資料檔案路徑
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """載入法說會事件資料"""
        print(f"載入事件資料: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"總共載入 {len(self.df)} 筆事件")
        print(f"欄位: {list(self.df.columns)}")
        return self

    def parse_date(self):
        """解析日期欄位 - 事件日T=0"""
        print("\n解析日期...")
        # 將 mdate 轉換為 datetime（法說會日期 = 事件日 T=0）
        self.df['mdate'] = pd.to_datetime(self.df['mdate'])
        print(f"日期範圍: {self.df['mdate'].min()} 至 {self.df['mdate'].max()}")
        return self

    def filter_events(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        過濾事件日期範圍

        Parameters:
        -----------
        start_date : str
            開始日期
        end_date : str
            結束日期
        """
        print(f"\n過濾事件日期: {start_date} 至 {end_date}")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        original_count = len(self.df)
        self.df = self.df[(self.df['mdate'] >= start) & (self.df['mdate'] <= end)].copy()
        filtered_count = len(self.df)

        print(f"過濾前: {original_count} 筆")
        print(f"過濾後: {filtered_count} 筆")
        return self

    def create_x1_variable(self):
        """
        建立X1虛擬變數：法說會是否提供公開錄影檔

        根據claude.md定義：
        - X1 = 1: weblink以"http"開頭（提供線上影音檔）
        - X1 = 0: 其他情況（空值、"線上法說會"、"電話會議"等）

        經濟意涵：衡量資訊透明度的影響
        """
        print("\n建立X1虛擬變數（是否提供公開錄影檔）...")

        # 處理weblink欄位
        weblink_series = self.df['weblink'].fillna('').astype(str)

        # weblink以http開頭表示提供線上影音檔
        self.df['X1'] = weblink_series.str.startswith('http', na=False).astype(int)

        # 統計
        x1_counts = self.df['X1'].value_counts()
        print(f"X1=0 (無錄影): {x1_counts.get(0, 0)} 筆")
        print(f"X1=1 (有錄影): {x1_counts.get(1, 0)} 筆")

        return self

    def save_processed_data(self, output_path='data/processed/event_list.csv'):
        """
        儲存處理後的事件列表

        Parameters:
        -----------
        output_path : str
            輸出檔案路徑
        """
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 選擇關鍵欄位：股票代號、事件日期、X1變數
        columns_to_save = ['coid', 'mdate', 'X1']

        self.df[columns_to_save].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n處理後的事件列表已儲存至: {output_path}")
        print(f"共 {len(self.df)} 筆事件")

        return output_path

    def get_summary_statistics(self):
        """取得摘要統計"""
        print("\n=== 摘要統計 ===")
        print(f"總事件數: {len(self.df)}")
        print(f"涵蓋股票數: {self.df['coid'].nunique()}")

        print(f"\n各股票事件次數統計:")
        event_counts = self.df['coid'].value_counts()
        print(f"  最多事件: {event_counts.max()} 次 (股票代號: {event_counts.idxmax()})")
        print(f"  平均事件: {event_counts.mean():.2f} 次")
        print(f"  中位數: {event_counts.median():.0f} 次")

        print(f"\nX1變數分布（資訊透明度）:")
        print(self.df['X1'].value_counts().sort_index())

        return event_counts

    def process_all(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        執行完整的事件分類流程

        Parameters:
        -----------
        start_date : str
            開始日期
        end_date : str
            結束日期

        Returns:
        --------
        str : 輸出檔案路徑
        """
        self.load_data()
        self.parse_date()
        self.filter_events(start_date, end_date)
        self.create_x1_variable()
        self.get_summary_statistics()
        output_path = self.save_processed_data()

        return output_path


def main():
    """主程式"""
    print("="*80)
    print("階段1：事件識別與分類（X1變數）")
    print("="*80)

    # 建立分類器並執行處理
    classifier = EventClassifier()
    output_path = classifier.process_all(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("\n" + "="*80)
    print("階段1完成！")
    print(f"輸出檔案: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
