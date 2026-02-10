#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段 9：敘述統計分析
計算所有變數的敘述統計量（樣本數、平均數、標準差、最小值、最大值）
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DescriptiveStatistics:
    """敘述統計分析器"""

    def __init__(self):
        """初始化"""
        self.project_root = self._get_project_root()
        self.merged_data = None
        self.statistics = None

    def _get_project_root(self):
        """取得專案根目錄"""
        current = Path.cwd()
        if current.name == 'src':
            return current.parent
        return current

    def load_and_merge_data(self):
        """載入並合併所有資料"""
        print("\n載入資料檔案...")

        # 載入各階段資料
        try:
            # Y: CAR 資料
            car_df = pd.read_csv(self.project_root / 'data/processed/car_data.csv')
            # CAR 資料使用 event_date 欄位
            if 'event_date' in car_df.columns:
                car_df['event_date'] = pd.to_datetime(car_df['event_date'])
                car_df = car_df.rename(columns={'event_date': 'mdate'})
            # 重新命名 CAR 欄位為 Y_CAR
            car_columns = [col for col in car_df.columns if col.startswith('CAR_')]
            if car_columns:
                car_df = car_df.rename(columns={car_columns[0]: 'Y_CAR'})
            print(f"  ✓ CAR資料: {len(car_df)} 筆")

            # X1: 事件分類
            events_df = pd.read_csv(self.project_root / 'data/processed/event_list.csv')
            events_df['mdate'] = pd.to_datetime(events_df['mdate'])
            print(f"  ✓ 事件列表: {len(events_df)} 筆")

            # X2: 週轉率
            x2_df = None
            x2_path = self.project_root / 'data/processed/x2_turnover.csv'
            if x2_path.exists():
                x2_df = pd.read_csv(x2_path)
                # 統一日期欄位名稱
                if 'event_date' in x2_df.columns:
                    x2_df['event_date'] = pd.to_datetime(x2_df['event_date'])
                    x2_df = x2_df.rename(columns={'event_date': 'mdate'})
                # 重新命名變數欄位
                turnover_cols = [col for col in x2_df.columns if col.startswith('X2_')]
                if turnover_cols:
                    x2_df = x2_df.rename(columns={turnover_cols[0]: 'X2_avg_turnover'})
                print(f"  ✓ X2週轉率: {len(x2_df)} 筆")

            # X3: 報酬率
            x3_df = None
            x3_path = self.project_root / 'data/processed/x3_returns.csv'
            if x3_path.exists():
                x3_df = pd.read_csv(x3_path)
                # 統一日期欄位名稱
                if 'event_date' in x3_df.columns:
                    x3_df['event_date'] = pd.to_datetime(x3_df['event_date'])
                    x3_df = x3_df.rename(columns={'event_date': 'mdate'})
                # 重新命名變數欄位
                return_cols = [col for col in x3_df.columns if col.startswith('X3_')]
                if return_cols:
                    x3_df = x3_df.rename(columns={
                        return_cols[0]: 'X3_ret_before'  # 假設是累積報酬
                    })
                    # 如果有第二個欄位，當作 ret_after
                    if len(return_cols) > 1:
                        x3_df = x3_df.rename(columns={return_cols[1]: 'X3_ret_after'})
                print(f"  ✓ X3報酬率: {len(x3_df)} 筆")

            # X4: 內部人交易 & X9: 散戶持股
            x4_df = None
            x4_path = self.project_root / 'data/processed/x4_insider.csv'
            if x4_path.exists():
                x4_df = pd.read_csv(x4_path)
                # 統一日期欄位名稱
                if 'event_date' in x4_df.columns:
                    x4_df['event_date'] = pd.to_datetime(x4_df['event_date'])
                    x4_df = x4_df.rename(columns={'event_date': 'mdate'})
                # 重新命名變數欄位
                insider_cols = [col for col in x4_df.columns if col.startswith('X4_')]
                retail_cols = [col for col in x4_df.columns if col.startswith('X9_')]
                if insider_cols:
                    x4_df = x4_df.rename(columns={insider_cols[0]: 'X4_insider_net_buy'})
                if retail_cols:
                    x4_df = x4_df.rename(columns={retail_cols[0]: 'X9_retail_holding_pct'})
                print(f"  ✓ X4內部人&X9散戶: {len(x4_df)} 筆")

            # X5: 距離前次法說會天數
            x5_df = None
            x5_path = self.project_root / 'data/processed/x5_days_since_last.csv'
            if x5_path.exists():
                x5_df = pd.read_csv(x5_path)
                # X5 已經使用 mdate
                if 'mdate' in x5_df.columns:
                    x5_df['mdate'] = pd.to_datetime(x5_df['mdate'])
                elif 'event_date' in x5_df.columns:
                    x5_df['event_date'] = pd.to_datetime(x5_df['event_date'])
                    x5_df = x5_df.rename(columns={'event_date': 'mdate'})
                print(f"  ✓ X5天數差: {len(x5_df)} 筆")

            # X6/X7/X8: 控制變數
            controls_df = None
            controls_path = self.project_root / 'data/processed/x6_x7_x8_controls.csv'
            if controls_path.exists():
                controls_df = pd.read_csv(controls_path)
                # 統一日期欄位名稱
                if 'event_date' in controls_df.columns:
                    controls_df['event_date'] = pd.to_datetime(controls_df['event_date'])
                    controls_df = controls_df.rename(columns={'event_date': 'mdate'})
                # 重新命名變數欄位
                size_cols = [col for col in controls_df.columns if col.startswith('X6_')]
                bm_cols = [col for col in controls_df.columns if col.startswith('X7_')]
                if size_cols:
                    controls_df = controls_df.rename(columns={size_cols[0]: 'X6_log_market_cap'})
                if bm_cols:
                    controls_df = controls_df.rename(columns={bm_cols[0]: 'X7_book_to_market'})
                print(f"  ✓ X6/X7/X8控制變數: {len(controls_df)} 筆")

        except Exception as e:
            print(f"錯誤：無法載入資料 - {str(e)}")
            raise

        # 合併資料
        print("\n合併資料...")
        merged = car_df.copy()

        # 合併 X1 (事件分類)
        x1_col = 'X1_has_video' if 'X1_has_video' in events_df.columns else 'X1'
        merged = merged.merge(
            events_df[['coid', 'mdate', x1_col]],
            on=['coid', 'mdate'],
            how='left'
        )
        # 統一欄位名稱
        if x1_col == 'X1':
            merged = merged.rename(columns={'X1': 'X1_has_video'})

        # 合併 X2
        if x2_df is not None:
            merged = merged.merge(
                x2_df[['coid', 'mdate', 'X2_avg_turnover']],
                on=['coid', 'mdate'],
                how='left'
            )

        # 合併 X3
        if x3_df is not None:
            # 只合併存在的欄位
            x3_cols = ['coid', 'mdate']
            if 'X3_ret_before' in x3_df.columns:
                x3_cols.append('X3_ret_before')
            if 'X3_ret_after' in x3_df.columns:
                x3_cols.append('X3_ret_after')
            
            merged = merged.merge(
                x3_df[x3_cols],
                on=['coid', 'mdate'],
                how='left'
            )

        # 合併 X4 & X6（持股變化率）
        if x4_df is not None:
            cols_to_merge = ['coid', 'mdate']
            if 'X4_top10_change_rate' in x4_df.columns:
                cols_to_merge.append('X4_top10_change_rate')
            elif 'X4_insider_net_buy' in x4_df.columns:
                cols_to_merge.append('X4_insider_net_buy')
            
            if 'X6_retail_change_rate' in x4_df.columns:
                cols_to_merge.append('X6_retail_change_rate')
            elif 'X9_retail_holding_pct' in x4_df.columns:
                cols_to_merge.append('X9_retail_holding_pct')
            
            merged = merged.merge(
                x4_df[cols_to_merge],
                on=['coid', 'mdate'],
                how='left'
            )

        # 合併 X5（支援對數轉換）
        if x5_df is not None:
            x5_col = 'X5_log_days_since_last' if 'X5_log_days_since_last' in x5_df.columns else 'X5_days_since_last'
            cols_to_merge = ['coid', 'mdate', x5_col]
            merged = merged.merge(
                x5_df[cols_to_merge],
                on=['coid', 'mdate'],
                how='left'
            )
            # 統一欄位名稱
            if x5_col == 'X5_log_days_since_last':
                merged['X5_days_since_last'] = merged[x5_col]
                merged = merged.drop(columns=[x5_col], errors='ignore')

        # 合併 X7/X8/X9（控制變數）
        if controls_df is not None:
            cols_to_merge = ['coid', 'mdate']
            # 支援新舊欄位名稱
            if 'X6_log_size' in controls_df.columns:
                cols_to_merge.append('X6_log_size')
            elif 'X6_log_market_cap' in controls_df.columns:
                cols_to_merge.append('X6_log_market_cap')
            
            if 'X7_bm_ratio' in controls_df.columns:
                cols_to_merge.append('X7_bm_ratio')
            elif 'X7_book_to_market' in controls_df.columns:
                cols_to_merge.append('X7_book_to_market')
            
            if 'X8_industry' in controls_df.columns:
                cols_to_merge.append('X8_industry')
            
            merged = merged.merge(
                controls_df[cols_to_merge],
                on=['coid', 'mdate'],
                how='left'
            )

        print(f"合併後資料: {len(merged)} 筆")
        print(f"欄位數: {len(merged.columns)}")

        self.merged_data = merged
        return merged

    def calculate_statistics(self):
        """計算敘述統計"""
        if self.merged_data is None:
            raise ValueError("請先執行 load_and_merge_data()")

        print("\n計算敘述統計...")

        # 定義要分析的變數
        variables = {
            'Y_CAR': 'CAR (應變數)',
            'X1_has_video': 'X1: 有錄影檔 (1=是, 0=否)',
            'X2_avg_turnover': 'X2: 平均週轉率 (%)',
            'X3_ret_before': 'X3: 事件前報酬率 (%)',
            'X3_ret_after': 'X3: 事件後報酬率 (%)',
            'X4_insider_net_buy': 'X4: 內部人淨買賣比率 (%)',
            'X5_days_since_last': 'X5: 距離前次法說會天數',
            'X6_log_market_cap': 'X6: 公司規模 (log市值)',
            'X7_book_to_market': 'X7: 淨值市價比',
            'X9_retail_holding_pct': 'X9: 散戶持股比例 (%)',
        }

        # 計算統計量
        stats_list = []

        for var_name, var_desc in variables.items():
            if var_name not in self.merged_data.columns:
                print(f"  ⚠ 跳過 {var_name}（欄位不存在）")
                continue

            data = self.merged_data[var_name].dropna()

            if len(data) == 0:
                print(f"  ⚠ 跳過 {var_name}（無有效數據）")
                continue

            stats = {
                '變數代號': var_name,
                '變數說明': var_desc,
                '樣本數': len(data),
                '平均數': data.mean(),
                '標準差': data.std(),
                '最小值': data.min(),
                '最大值': data.max(),
                '中位數': data.median(),
                '第25百分位': data.quantile(0.25),
                '第75百分位': data.quantile(0.75),
            }

            stats_list.append(stats)
            print(f"  ✓ {var_name}: N={len(data)}, Mean={data.mean():.4f}, SD={data.std():.4f}")

        # 轉換為 DataFrame
        self.statistics = pd.DataFrame(stats_list)

        # 添加產業分布統計（X8）
        if 'X8_industry' in self.merged_data.columns:
            print("\n計算產業分布...")
            industry_counts = self.merged_data['X8_industry'].value_counts()
            print(f"  產業類別數: {len(industry_counts)}")
            print("  前5大產業:")
            for ind, count in industry_counts.head(5).items():
                pct = count / len(self.merged_data) * 100
                print(f"    {ind}: {count} 筆 ({pct:.1f}%)")

        return self.statistics

    def save_results(self):
        """儲存結果"""
        if self.statistics is None:
            raise ValueError("請先執行 calculate_statistics()")

        # 確保輸出目錄存在
        output_dir = self.project_root / 'results'
        output_dir.mkdir(exist_ok=True)

        # 儲存敘述統計表
        output_path = output_dir / 'descriptive_statistics.csv'
        self.statistics.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 敘述統計表已儲存: {output_path}")

        # 同時儲存一份更易讀的版本（四捨五入到小數點後4位）
        stats_rounded = self.statistics.copy()
        numeric_cols = ['平均數', '標準差', '最小值', '最大值', '中位數', '第25百分位', '第75百分位']
        for col in numeric_cols:
            if col in stats_rounded.columns:
                stats_rounded[col] = stats_rounded[col].round(4)

        output_path_rounded = output_dir / 'descriptive_statistics_rounded.csv'
        stats_rounded.to_csv(output_path_rounded, index=False, encoding='utf-8-sig')
        print(f"✓ 敘述統計表(四捨五入)已儲存: {output_path_rounded}")

        # 產生摘要報告
        self._generate_summary_report()

        return str(output_path)

    def _generate_summary_report(self):
        """產生摘要報告"""
        output_dir = self.project_root / 'results'
        report_path = output_dir / 'descriptive_statistics_summary.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("敘述統計分析摘要報告\n")
            f.write("="*80 + "\n\n")

            # 基本資訊
            f.write(f"分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總樣本數: {len(self.merged_data)} 筆事件\n\n")

            # 變數統計
            f.write("="*80 + "\n")
            f.write("變數敘述統計\n")
            f.write("="*80 + "\n\n")

            for _, row in self.statistics.iterrows():
                f.write(f"{row['變數代號']}: {row['變數說明']}\n")
                f.write(f"  樣本數: {int(row['樣本數']):,}\n")
                f.write(f"  平均數: {row['平均數']:.4f}\n")
                f.write(f"  標準差: {row['標準差']:.4f}\n")
                f.write(f"  最小值: {row['最小值']:.4f}\n")
                f.write(f"  最大值: {row['最大值']:.4f}\n")
                f.write(f"  中位數: {row['中位數']:.4f}\n")
                f.write(f"  25%分位: {row['第25百分位']:.4f}\n")
                f.write(f"  75%分位: {row['第75百分位']:.4f}\n")
                f.write("\n")

            # X1 分布（類別變數）
            if 'X1_has_video' in self.merged_data.columns:
                f.write("="*80 + "\n")
                f.write("X1: 錄影檔分布\n")
                f.write("="*80 + "\n\n")
                x1_counts = self.merged_data['X1_has_video'].value_counts()
                total = len(self.merged_data)
                for val, count in x1_counts.items():
                    pct = count / total * 100
                    label = "有錄影檔" if val == 1 else "無錄影檔"
                    f.write(f"  {label}: {count:,} 筆 ({pct:.1f}%)\n")
                f.write("\n")

            # X8 產業分布
            if 'X8_industry' in self.merged_data.columns:
                f.write("="*80 + "\n")
                f.write("X8: 產業分布（前10大）\n")
                f.write("="*80 + "\n\n")
                industry_counts = self.merged_data['X8_industry'].value_counts().head(10)
                total = len(self.merged_data)
                for ind, count in industry_counts.items():
                    pct = count / total * 100
                    f.write(f"  {ind}: {count:,} 筆 ({pct:.1f}%)\n")
                f.write(f"\n  總產業類別數: {self.merged_data['X8_industry'].nunique()}\n\n")

            # 資料完整性報告
            f.write("="*80 + "\n")
            f.write("資料完整性\n")
            f.write("="*80 + "\n\n")

            variables = ['Y_CAR', 'X1_has_video', 'X2_avg_turnover', 'X3_ret_before', 
                        'X3_ret_after', 'X4_insider_net_buy', 'X5_days_since_last',
                        'X6_log_market_cap', 'X7_book_to_market', 'X9_retail_holding_pct']

            total_obs = len(self.merged_data)
            for var in variables:
                if var in self.merged_data.columns:
                    missing = self.merged_data[var].isna().sum()
                    available = total_obs - missing
                    pct = available / total_obs * 100
                    f.write(f"  {var}: {available:,} / {total_obs:,} ({pct:.1f}%)\n")

        print(f"✓ 摘要報告已儲存: {report_path}")


def main():
    """測試用主程式"""
    analyzer = DescriptiveStatistics()
    analyzer.load_and_merge_data()
    analyzer.calculate_statistics()
    analyzer.save_results()
    print("\n敘述統計分析完成！")


if __name__ == '__main__':
    main()
