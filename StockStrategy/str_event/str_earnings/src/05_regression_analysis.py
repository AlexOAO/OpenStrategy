#!/usr/bin/env python3
"""
階段 5：迴歸分析
執行多方和空方模型的迴歸分析
"""
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns


class RegressionAnalyzer:
    """迴歸分析器"""

    def __init__(
        self,
        car_data_path='data/processed/car_data.csv',
        long_signals_path='data/processed/long_signals.csv',
        short_signals_path='data/processed/short_signals.csv',
        event_list_path='data/processed/event_list.csv'
    ):
        """
        初始化分析器

        Parameters:
        -----------
        car_data_path : str
            CAR 資料檔案路徑
        long_signals_path : str
            多方訊號檔案路徑
        short_signals_path : str
            空方訊號檔案路徑
        event_list_path : str
            事件列表檔案路徑
        """
        self.car_data_path = car_data_path
        self.long_signals_path = long_signals_path
        self.short_signals_path = short_signals_path
        self.event_list_path = event_list_path

        self.merged_data = None
        self.long_model_results = {}
        self.short_model_results = {}
        self.long_model_descriptive_stats = {}
        self.short_model_descriptive_stats = {}
        self.long_model_summaries = {}
        self.short_model_summaries = {}

    def load_and_merge_data(self):
        """載入並合併所有資料"""
        print("載入資料...")

        # 載入各階段資料
        car_df = pd.read_csv(self.car_data_path)
        event_df = pd.read_csv(self.event_list_path)

        # 嘗試載入訊號資料
        try:
            long_df = pd.read_csv(self.long_signals_path)
            has_long = True
        except:
            print("警告：找不到多方訊號資料")
            long_df = pd.DataFrame()
            has_long = False

        try:
            short_df = pd.read_csv(self.short_signals_path)
            has_short = True
        except:
            print("警告：找不到空方訊號資料")
            short_df = pd.DataFrame()
            has_short = False

        # 轉換日期格式
        for df in [car_df, event_df, long_df, short_df]:
            if 'mdate' in df.columns:
                df['mdate'] = pd.to_datetime(df['mdate'])

        # 合併資料
        merged = car_df.copy()

        # 合併事件資料（錄影虛擬變數）
        if 'X1' not in merged.columns:
            columns = ['coid', 'mdate', 'X1']
            if 'meeting_type' in event_df.columns:
                columns.append('meeting_type')

            merged = merged.merge(
                event_df[columns],
                on=['coid', 'mdate'],
                how='left'
            )

        # 合併多方訊號
        if has_long and len(long_df) > 0:
            merged = merged.merge(
                long_df,
                on=['coid', 'mdate'],
                how='left',
                suffixes=('', '_long')
            )

        # 合併空方訊號
        if has_short and len(short_df) > 0:
            merged = merged.merge(
                short_df,
                on=['coid', 'mdate'],
                how='left',
                suffixes=('', '_short')
            )

        self.merged_data = merged
        print(f"合併後資料: {len(merged)} 筆")
        print(f"可用欄位: {list(merged.columns)}")

        return self

    def prepare_data_for_regression(self, y_var, x_vars, standardize=True):
        """
        準備迴歸分析資料

        Parameters:
        -----------
        y_var : str
            應變數名稱
        x_vars : list
            自變數名稱列表
        standardize : bool
            是否標準化連續變數（錄影虛擬變數除外）

        Returns:
        --------
        tuple : (y, X) - 應變數和自變數 DataFrame
        """
        # 檢查應變數是否存在
        if y_var not in self.merged_data.columns:
            print(f"錯誤：應變數 {y_var} 不存在")
            return None, None, None, None

        # 檢查每個自變數的缺失比例，排除缺失過多的變數（>80%缺失）
        x_vars_filtered = []
        for var in x_vars:
            if var in self.merged_data.columns:
                missing_pct = self.merged_data[var].isna().sum() / len(self.merged_data)
                if missing_pct <= 0.8:
                    x_vars_filtered.append(var)
                else:
                    print(f"排除變數 {var}（缺失率: {missing_pct:.1%}）")

        if len(x_vars_filtered) == 0:
            print("錯誤：沒有可用的自變數")
            return None, None, None, None

        print(f"使用的自變數: {x_vars_filtered}")

        # 選擇所需欄位並過濾缺失值
        vars_to_use = [y_var] + x_vars_filtered
        sample = self.merged_data[vars_to_use].dropna()

        if len(sample) == 0:
            print("錯誤：沒有有效樣本")
            return None, None, None, None

        print(f"有效樣本數: {len(sample)}")

        # 準備應變數
        y = sample[y_var]

        # 準備自變數
        X = sample[x_vars_filtered].copy()

        # 蒐集統計指標（使用原始數值避免標準化後失去意義）
        stats = OrderedDict()
        for col in vars_to_use:
            series = sample[col]
            stats[col] = {
                'mean': float(series.mean()) if len(series) > 0 else np.nan,
                'median': float(series.median()) if len(series) > 0 else np.nan,
                'std': float(series.std(ddof=1)) if len(series) > 1 else 0.0
            }

        # 標準化連續變數（排除虛擬變數）
        if standardize:
            dummy_vars = ['X1']
            continuous_vars = [v for v in x_vars_filtered if v not in dummy_vars]

            if continuous_vars:
                print(f"標準化變數: {continuous_vars}")
                for var in continuous_vars:
                    mean = X[var].mean()
                    std = X[var].std()
                    if std > 0:  # 避免除以零
                        X[var] = (X[var] - mean) / std
                    else:
                        print(f"警告：{var} 標準差為0，跳過標準化")

        # 添加常數項
        X = sm.add_constant(X)

        return y, X, stats, len(sample)

    def run_long_model(self, car_var='CAR_0_p2'):
        """
        執行多方模型迴歸（完整版）

        Parameters:
        -----------
        car_var : str
            CAR 變數名稱

        Returns:
        --------
        statsmodels結果物件
        """
        print("\n" + "="*60)
        print(f"多方模型迴歸分析 (應變數: {car_var})")
        print("="*60)

        # 定義自變數（完整版 - 5個交易訊號）
        x_vars = [
            'X1',
            'institutional_buying',
            'insider_net_buying',
            'historical_high_volume',
            'margin_purchase_increase',
            'short_covering'
        ]

        # 準備資料
        y, X, stats, sample_size = self.prepare_data_for_regression(car_var, x_vars)

        if y is None:
            return None

        # 執行迴歸
        model = sm.OLS(y, X)
        results = model.fit()

        # 輸出結果
        summary_text = results.summary().as_text()
        print("\n" + summary_text)

        # 儲存結果
        self.long_model_results[car_var] = results
        self.long_model_descriptive_stats[car_var] = {
            'sample_size': sample_size,
            'stats': stats
        }
        self.long_model_summaries[car_var] = summary_text

        return results

    def run_short_model(self, car_var='CAR_0_p2'):
        """
        執行空方模型迴歸（完整版）

        Parameters:
        -----------
        car_var : str
            CAR 變數名稱

        Returns:
        --------
        statsmodels結果物件
        """
        print("\n" + "="*60)
        print(f"空方模型迴歸分析 (應變數: {car_var})")
        print("="*60)

        # 定義自變數（完整版 - 5個交易訊號）
        x_vars = [
            'X1',
            'ashvol',
            'short_interest_increase',
            'insider_net_selling',
            'retail_net_buying',
            'caid'
        ]

        # 準備資料
        y, X, stats, sample_size = self.prepare_data_for_regression(car_var, x_vars)

        if y is None:
            return None

        # 執行迴歸
        model = sm.OLS(y, X)
        results = model.fit()

        # 輸出結果
        summary_text = results.summary().as_text()
        print("\n" + summary_text)

        # 儲存結果
        self.short_model_results[car_var] = results
        self.short_model_descriptive_stats[car_var] = {
            'sample_size': sample_size,
            'stats': stats
        }
        self.short_model_summaries[car_var] = summary_text

        return results

    def run_all_regressions(self):
        """執行所有迴歸分析"""
        car_vars = ['CAR_m1_p1', 'CAR_0_p2', 'CAR_0_p4']

        print("\n" + "="*60)
        print("執行多方模型迴歸")
        print("="*60)

        for car_var in car_vars:
            if car_var in self.merged_data.columns:
                self.run_long_model(car_var)

        print("\n" + "="*60)
        print("執行空方模型迴歸")
        print("="*60)

        for car_var in car_vars:
            if car_var in self.merged_data.columns:
                self.run_short_model(car_var)

    def save_regression_results(self, output_dir='results'):
        """
        儲存迴歸結果

        Parameters:
        -----------
        output_dir : str
            輸出目錄
        """
        os.makedirs(output_dir, exist_ok=True)

        # 儲存多方模型結果
        if self.long_model_results:
            long_results_list = []
            for car_var, results in self.long_model_results.items():
                # 提取係數、標準誤、t值、p值
                summary_df = pd.DataFrame({
                    'variable': results.params.index,
                    'coefficient': results.params.values,
                    'std_err': results.bse.values,
                    't_value': results.tvalues.values,
                    'p_value': results.pvalues.values,
                    'conf_low': results.conf_int()[0].values,
                    'conf_high': results.conf_int()[1].values
                })
                summary_df['car_var'] = car_var
                summary_df['r_squared'] = results.rsquared
                summary_df['adj_r_squared'] = results.rsquared_adj
                summary_df['f_statistic'] = results.fvalue
                summary_df['f_pvalue'] = results.f_pvalue
                summary_df['n_obs'] = results.nobs

                long_results_list.append(summary_df)

            if long_results_list:
                long_results_df = pd.concat(long_results_list, ignore_index=True)
                long_path = os.path.join(output_dir, 'long_model_results.csv')
                long_results_df.to_csv(long_path, index=False, encoding='utf-8-sig')
                print(f"\n多方模型結果已儲存至: {long_path}")

        # 儲存空方模型結果
        if self.short_model_results:
            short_results_list = []
            for car_var, results in self.short_model_results.items():
                summary_df = pd.DataFrame({
                    'variable': results.params.index,
                    'coefficient': results.params.values,
                    'std_err': results.bse.values,
                    't_value': results.tvalues.values,
                    'p_value': results.pvalues.values,
                    'conf_low': results.conf_int()[0].values,
                    'conf_high': results.conf_int()[1].values
                })
                summary_df['car_var'] = car_var
                summary_df['r_squared'] = results.rsquared
                summary_df['adj_r_squared'] = results.rsquared_adj
                summary_df['f_statistic'] = results.fvalue
                summary_df['f_pvalue'] = results.f_pvalue
                summary_df['n_obs'] = results.nobs

                short_results_list.append(summary_df)

            if short_results_list:
                short_results_df = pd.concat(short_results_list, ignore_index=True)
                short_path = os.path.join(output_dir, 'short_model_results.csv')
                short_results_df.to_csv(short_path, index=False, encoding='utf-8-sig')
                print(f"空方模型結果已儲存至: {short_path}")

        # 匯出 Markdown 報告
        md_path = os.path.join(output_dir, 'regression_report.md')
        self.save_markdown_report(md_path)

    def generate_summary_report(self, output_path='results/summary_report.txt'):
        """
        生成摘要報告

        Parameters:
        -----------
        output_path : str
            輸出檔案路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("法說會事件交易訊號分析 - 摘要報告\n")
            f.write("="*80 + "\n\n")

            # 資料概況
            f.write("## 資料概況\n\n")
            f.write(f"總事件數: {len(self.merged_data)}\n")
            f.write(f"涵蓋股票數: {self.merged_data['coid'].nunique()}\n\n")

            # 多方模型結果摘要
            if self.long_model_results:
                f.write("## 多方模型結果摘要\n\n")
                for car_var, results in self.long_model_results.items():
                    f.write(f"### {car_var}\n")
                    f.write(f"樣本數: {int(results.nobs)}\n")
                    f.write(f"R²: {results.rsquared:.4f}\n")
                    f.write(f"調整後 R²: {results.rsquared_adj:.4f}\n")
                    f.write(f"F統計量: {results.fvalue:.4f} (p-value: {results.f_pvalue:.4f})\n\n")

                    # 顯著變數
                    sig_vars = results.pvalues[results.pvalues < 0.05]
                    if len(sig_vars) > 0:
                        f.write("顯著變數 (p < 0.05):\n")
                        for var, pval in sig_vars.items():
                            coef = results.params[var]
                            f.write(f"  - {var}: 係數={coef:.6f}, p={pval:.4f}\n")
                    f.write("\n")

            # 空方模型結果摘要
            if self.short_model_results:
                f.write("## 空方模型結果摘要\n\n")
                for car_var, results in self.short_model_results.items():
                    f.write(f"### {car_var}\n")
                    f.write(f"樣本數: {int(results.nobs)}\n")
                    f.write(f"R²: {results.rsquared:.4f}\n")
                    f.write(f"調整後 R²: {results.rsquared_adj:.4f}\n")
                    f.write(f"F統計量: {results.fvalue:.4f} (p-value: {results.f_pvalue:.4f})\n\n")

                    # 顯著變數
                    sig_vars = results.pvalues[results.pvalues < 0.05]
                    if len(sig_vars) > 0:
                        f.write("顯著變數 (p < 0.05):\n")
                        for var, pval in sig_vars.items():
                            coef = results.params[var]
                            f.write(f"  - {var}: 係數={coef:.6f}, p={pval:.4f}\n")
                    f.write("\n")

        print(f"\n摘要報告已儲存至: {output_path}")

    def save_markdown_report(self, output_path='results/regression_report.md'):
        """輸出包含統計摘要與 OLS 結果的 Markdown 報告"""
        if not self.long_model_results and not self.short_model_results:
            print("警告：沒有迴歸結果可輸出 Markdown 報告")
            return None

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        def fmt(value):
            if value is None:
                return 'NaN'
            try:
                if pd.isna(value):
                    return 'NaN'
            except TypeError:
                pass
            return f"{float(value):.6f}"

        lines = ["# 迴歸分析結果報告", ""]

        def append_section(title, summaries, stats_dict):
            if not summaries:
                return

            for car_var, summary_text in summaries.items():
                lines.append(f"## {title} - {car_var}")

                desc = stats_dict.get(car_var, {})
                sample_size = desc.get('sample_size')
                if sample_size is not None:
                    lines.append(f"- 有效樣本數: {int(sample_size)}")

                var_stats = desc.get('stats', {})
                if var_stats:
                    lines.append("\n| 變數 | 平均 | 中位數 | 標準差 | 有效樣本數 |")
                    lines.append("| --- | --- | --- | --- | --- |")
                    for var_name, metrics in var_stats.items():
                        mean_val = fmt(metrics.get('mean'))
                        median_val = fmt(metrics.get('median'))
                        std_val = fmt(metrics.get('std'))
                        count_val = int(sample_size) if sample_size is not None else '-'
                        lines.append(
                            f"| {var_name} | {mean_val} | {median_val} | {std_val} | {count_val} |"
                        )

                if summary_text:
                    lines.append("\n```text")
                    lines.append(summary_text.strip())
                    lines.append("```")

                lines.append("")

        append_section('多方模型', self.long_model_summaries, self.long_model_descriptive_stats)
        append_section('空方模型', self.short_model_summaries, self.short_model_descriptive_stats)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines).rstrip() + "\n")

        print(f"Markdown 報告已儲存至: {output_path}")
        return output_path


def main():
    """主程式"""
    print("="*60)
    print("階段 5：迴歸分析")
    print("="*60)

    # 建立分析器
    analyzer = RegressionAnalyzer()

    # 載入並合併資料
    analyzer.load_and_merge_data()

    # 執行迴歸分析
    analyzer.run_all_regressions()

    # 儲存結果
    analyzer.save_regression_results()

    # 生成摘要報告
    analyzer.generate_summary_report()

    print("\n" + "="*60)
    print("階段 5 完成！")
    print("="*60)


if __name__ == '__main__':
    main()
