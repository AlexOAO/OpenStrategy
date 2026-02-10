#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段8：統一迴歸分析
整合所有變數（Y, X1-X9）並執行OLS迴歸
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


def get_project_root():
    current = Path.cwd()
    return current.parent if current.name == 'src' else current

PROJECT_ROOT = get_project_root()


class UnifiedRegressionAnalyzer:
    def __init__(self):
        self.merged_df = None
        self.regression_results = None
        self.industry_dummies_cols = []  # 儲存產業虛擬變數名稱

    def load_and_merge_data(self):
        """
        載入並合併所有階段的資料

        預期檔案：
        - car_data.csv: Y（CAR），事件日期，股票代號
        - event_list.csv: X1（錄影檔虛擬變數）
        - x2_turnover.csv: X2（20日累積週轉率）
        - x3_returns.csv: X3（10日累積報酬率）
        - x4_insider.csv: X4（十大股東持股變化率）、X6（散戶持股變化率）
        - x5_days_since_last.csv: X5（距離前次法說會天數的對數）
        - x6_x7_x8_controls.csv: X7（規模）、X8（B/M）、X9（產業）
        """
        print("="*80)
        print("階段8：統一迴歸分析")
        print("="*80)
        print("\n載入資料...")

        data_dir = PROJECT_ROOT / 'data/processed'

        # 載入Y（應變數）
        car_df = pd.read_csv(data_dir / 'car_data.csv')
        car_df['event_date'] = pd.to_datetime(car_df['event_date'])
        
        # 動態檢測CAR欄位名稱（支援不同的窗期參數）
        car_cols = [col for col in car_df.columns if col.startswith('CAR_')]
        if not car_cols:
            raise ValueError("找不到CAR欄位！car_data.csv中應包含以'CAR_'開頭的欄位")
        self.car_column = car_cols[0]  # 取第一個CAR欄位
        print(f"  CAR資料: {len(car_df)} 筆，使用欄位: {self.car_column}")

        # 載入X1（來自event_list）
        event_df = pd.read_csv(data_dir / 'event_list.csv')
        event_df['mdate'] = pd.to_datetime(event_df['mdate'])
        print(f"  X1資料: {len(event_df)} 筆")

        # 載入X2-X9
        x2_df = pd.read_csv(data_dir / 'x2_turnover.csv') if (data_dir / 'x2_turnover.csv').exists() else None
        x3_df = pd.read_csv(data_dir / 'x3_returns.csv') if (data_dir / 'x3_returns.csv').exists() else None
        x4_df = pd.read_csv(data_dir / 'x4_insider.csv') if (data_dir / 'x4_insider.csv').exists() else None
        x5_df = pd.read_csv(data_dir / 'x5_days_since_last.csv') if (data_dir / 'x5_days_since_last.csv').exists() else None
        x6x7x8_df = pd.read_csv(data_dir / 'x6_x7_x8_controls.csv') if (data_dir / 'x6_x7_x8_controls.csv').exists() else None

        # 合併資料（以CAR為基礎）
        merged = car_df.copy()
        merged = merged.rename(columns={'event_date': 'mdate'})

        # 合併X1
        merged = merged.merge(event_df[['coid', 'mdate', 'X1']], on=['coid', 'mdate'], how='left')

        # 合併X2（週轉率）
        if x2_df is not None:
            x2_df = x2_df.copy()
            x2_df['event_date'] = pd.to_datetime(x2_df['event_date'])
            # 只選取需要的欄位
            x2_cols = ['coid', 'event_date', 'X2_cumulative_turnover_20d']
            merged = merged.merge(x2_df[x2_cols], left_on=['coid', 'mdate'], right_on=['coid', 'event_date'], how='left')
            merged = merged.drop(columns=['event_date'], errors='ignore')  # 刪除重複的 event_date
            print(f"  X2資料: {len(x2_df)} 筆")

        # 合併X3（報酬率）
        if x3_df is not None:
            x3_df = x3_df.copy()
            x3_df['event_date'] = pd.to_datetime(x3_df['event_date'])
            # 動態檢測X3欄位名稱（支持10日或20日）
            x3_col = 'X3_cumulative_return_10d' if 'X3_cumulative_return_10d' in x3_df.columns else 'X3_cumulative_return_20d'
            x3_cols = ['coid', 'event_date', x3_col]
            merged = merged.merge(x3_df[x3_cols], left_on=['coid', 'mdate'], right_on=['coid', 'event_date'], how='left')
            # 統一欄位名稱為 X3_cumulative_return
            if x3_col in merged.columns:
                merged['X3_cumulative_return'] = merged[x3_col]
                merged = merged.drop(columns=[x3_col], errors='ignore')
            merged = merged.drop(columns=['event_date'], errors='ignore')
            print(f"  X3資料: {len(x3_df)} 筆（使用欄位: {x3_col}）")

        # 合併X4和X6（十大股東與散戶持股變化率）
        if x4_df is not None:
            x4_df = x4_df.copy()
            x4_df['event_date'] = pd.to_datetime(x4_df['event_date'])
            # X4 是十大股東持股變化率，X6 是散戶持股變化率
            x4_cols = ['coid', 'event_date', 'X4_top10_change_rate', 'X6_retail_change_rate']
            # 只選取存在的欄位
            available_x4_cols = [col for col in x4_cols if col in x4_df.columns or col in ['coid', 'event_date']]
            merged = merged.merge(x4_df[available_x4_cols], left_on=['coid', 'mdate'], right_on=['coid', 'event_date'], how='left')
            merged = merged.drop(columns=['event_date'], errors='ignore')
            print(f"  X4&X6資料: {len(x4_df)} 筆")

        # 合併X5（距離天數的對數）
        if x5_df is not None:
            x5_df = x5_df.copy()
            if 'mdate' in x5_df.columns:
                x5_df['mdate'] = pd.to_datetime(x5_df['mdate'])
                # 使用對數轉換後的欄位
                x5_col = 'X5_log_days_since_last' if 'X5_log_days_since_last' in x5_df.columns else 'X5_days_since_last'
                x5_cols = ['coid', 'mdate', x5_col]
                merged = merged.merge(x5_df[x5_cols], on=['coid', 'mdate'], how='left')
                # 統一欄位名稱為 X5_days_since_last
                if x5_col == 'X5_log_days_since_last':
                    merged['X5_days_since_last'] = merged[x5_col]
                    merged = merged.drop(columns=[x5_col], errors='ignore')
            print(f"  X5資料: {len(x5_df)} 筆（使用對數轉換）")

        # 合併X7、X8、X9（控制變數）
        if x6x7x8_df is not None:
            x6x7x8_df = x6x7x8_df.copy()
            x6x7x8_df['event_date'] = pd.to_datetime(x6x7x8_df['event_date'])
            # X7=規模, X8=B/M, X9=產業
            x789_cols = ['coid', 'event_date', 'X6_log_size', 'X7_bm_ratio', 'X8_industry']
            # 重新命名以符合新的變數編號
            merged = merged.merge(x6x7x8_df[x789_cols], left_on=['coid', 'mdate'], right_on=['coid', 'event_date'], how='left')
            # 將欄位重新命名
            merged = merged.rename(columns={
                'X6_log_size': 'X7_log_size',
                'X7_bm_ratio': 'X8_bm_ratio',
                'X8_industry': 'X9_industry'
            })
            merged = merged.drop(columns=['event_date'], errors='ignore')
            print(f"  X7/X8/X9資料: {len(x6x7x8_df)} 筆")

        self.merged_df = merged
        print(f"\n合併後資料: {len(self.merged_df)} 筆")
        print(f"可用欄位: {list(self.merged_df.columns)}\n")

        # 顯示每個變數的資料完整性
        print("變數資料完整性檢查:")
        key_vars = [self.car_column, 'X1', 'X2_cumulative_turnover_20d', 'X3_cumulative_return',
                    'X4_top10_change_rate', 'X5_days_since_last', 'X6_retail_change_rate',
                    'X7_log_size', 'X8_bm_ratio', 'X9_industry']
        for var in key_vars:
            if var in self.merged_df.columns:
                non_null = self.merged_df[var].notna().sum()
                print(f"  {var}: {non_null}/{len(self.merged_df)} ({non_null/len(self.merged_df)*100:.1f}%)")
            else:
                print(f"  {var}: 欄位不存在")

        # 移除缺失值策略：只要 X1-X9 任一變數有值就保留（inner join 改為 outer join的概念）
        # 但至少 CAR 和 X1 必須存在
        original_count = len(self.merged_df)
        self.merged_df = self.merged_df[self.merged_df[self.car_column].notna() & self.merged_df['X1'].notna()]
        print(f"移除CAR或X1缺失後: {len(self.merged_df)} 筆（減少 {original_count - len(self.merged_df)} 筆）")
        
        # ★★★ 去除重複的 (coid, mdate) ★★★
        before_dedup = len(self.merged_df)
        self.merged_df = self.merged_df.drop_duplicates(subset=['coid', 'mdate'], keep='first')
        print(f"去除重複事件後: {len(self.merged_df)} 筆（減少 {before_dedup - len(self.merged_df)} 筆）\n")

        if len(self.merged_df) == 0:
            print("警告：移除缺失值後無可用樣本！")
            print("提示：請先執行階段2-7以產生足夠的變數資料\n")

        return self

    def run_regression(self):
        """
        執行OLS迴歸

        模型：CAR = α + β1×X1 + β2×X2 + ... + β9×X9 + ε
        """
        print("="*80)
        print("執行OLS迴歸")
        print("="*80)

        if len(self.merged_df) == 0:
            print("錯誤：無可用樣本，無法執行迴歸\n")
            return self

        # 準備Y和X
        y = self.merged_df[self.car_column]  # 應變數（動態CAR欄位）

        # 自變數列表
        x_vars = []
        available_vars = []

        var_mapping = {
            'X1': 'X1',
            'X2_cumulative_turnover_20d': 'X2',
            'X3_cumulative_return': 'X3',
            'X4_top10_change_rate': 'X4',
            'X5_days_since_last': 'X5',
            'X6_retail_change_rate': 'X6',
            'X7_log_size': 'X7',
            'X8_bm_ratio': 'X8',
            'X9_industry': 'X9'  # 產業虛擬變數
        }

        for col in self.merged_df.columns:
            if col in var_mapping:
                # 檢查該變數是否有至少一些非缺失值
                non_null_count = self.merged_df[col].notna().sum()
                total_count = len(self.merged_df)
                if non_null_count > 0:
                    x_vars.append(col)
                    available_vars.append(var_mapping[col])
                    print(f"  ✓ {var_mapping[col]} ({col}): {non_null_count}/{total_count} 非缺失")
                else:
                    print(f"  ✗ {var_mapping[col]} ({col}): 全部缺失，跳過")

        if len(x_vars) == 0:
            print("錯誤：沒有可用的自變數\n")
            return self

        print(f"\n選定的自變數: {available_vars}\n")

        X = self.merged_df[x_vars].copy()

        # 移除有任何NaN的行
        valid_indices = X.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) == 0:
            print(f"錯誤：所有樣本在自變數 {available_vars} 上都有缺失值\n")
            return self

        # 處理資料型別
        print("檢查資料型別...")
        for col in X.columns:
            print(f"  {col}: {X[col].dtype}")

        # 處理X9_industry（類別變數） - 拆分為虛擬變數 (Dummy Variables)
        if 'X9_industry' in X.columns:
            print("\n處理產業變數...")
            # 統計產業分佈
            industry_counts = X['X9_industry'].value_counts()
            print(f"  產業類別數: {len(industry_counts)}")
            print(f"  最常見產業: {industry_counts.head(5).to_dict()}")

            # ★★★ 使用虛擬變數編碼 (Dummy Encoding) ★★★
            # drop_first=True: 避免完全共線性（參考組為第一個類別）
            industry_dummies = pd.get_dummies(X['X9_industry'], prefix='X9', drop_first=True)
            self.industry_dummies_cols = industry_dummies.columns.tolist()
            
            print(f"  已轉換為 {len(self.industry_dummies_cols)} 個虛擬變數（drop_first=True）")
            print(f"  參考組（基準）: {sorted(X['X9_industry'].unique())[0]}")
            print(f"  虛擬變數範例: {self.industry_dummies_cols[:3]}{'...' if len(self.industry_dummies_cols) > 3 else ''}")
            
            # 移除原始 X9_industry 欄位，加入虛擬變數
            X = X.drop(columns=['X9_industry'])
            X = pd.concat([X, industry_dummies], axis=1)

        # 確保其他列都是數值型別
        print("\n轉換變數為數值型別...")
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                print(f"  警告: {col} 仍是 {X[col].dtype}，強制轉換")
                X[col] = pd.to_numeric(X[col], errors='coerce')
            else:
                # 確保是數值型別
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print("\n轉換後的資料型別:")
        for col in X.columns:
            print(f"  {col}: {X[col].dtype}, 缺失值: {X[col].isna().sum()}")
        
        # 再次移除因轉換產生的NaN
        valid_indices = X.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            print(f"錯誤：轉換為數值後所有樣本都有缺失值\n")
            return self

        # 最後確認：確保所有數據都是float64
        X = X.astype(float)
        y = y.astype(float)

        # 按照 X1-X8 順序重新排列欄位（在加入 const 之前）
        # X9（產業虛擬變數）已拆分，不包含在主要變數中
        desired_order = [
            'X1',
            'X2_cumulative_turnover_20d',
            'X3_cumulative_return',
            'X4_top10_change_rate',
            'X5_days_since_last',
            'X6_retail_change_rate',
            'X7_log_size',
            'X8_bm_ratio'
        ]
        
        # 只保留實際存在的主要變數
        ordered_cols = [col for col in desired_order if col in X.columns]
        
        # 產業虛擬變數放在最後
        X = X[ordered_cols + self.industry_dummies_cols]

        X = sm.add_constant(X)  # 加入截距項

        print(f"\n應變數: {self.car_column}")
        main_vars_display = [v for v in available_vars if v != 'X9']  # X9 是產業虛擬變數
        print(f"主要自變數: {main_vars_display}")
        if self.industry_dummies_cols:
            print(f"產業虛擬變數 (X9): {len(self.industry_dummies_cols)} 個")
        print(f"樣本數: {len(X)}")
        print(f"變數個數: {len(X.columns)}")
        
        # 檢查樣本數是否足夠
        if len(X) < len(X.columns) + 5:
            print(f"\n警告：樣本數({len(X)})相對變數個數({len(X.columns)})太少！")
            print("建議增加 --sample 參數或移除部分變數\n")

        # 執行OLS迴歸
        model = sm.OLS(y, X)
        results = model.fit()

        # 印出完整結果（包含所有產業虛擬變數）
        print("\n" + "="*80)
        print("迴歸結果（完整 - 包含所有產業虛擬變數）")
        print("="*80)
        print(results.summary())
        print("\n")

        # 印出主要變數摘要（X1-X8，不含產業虛擬變數）
        print("="*80)
        print("主要變數係數摘要（X1-X8）")
        print("="*80)

        print(f"{'變數':<30} {'係數':>12} {'標準誤':>12} {'t值':>10} {'P>|t|':>10} {'顯著性':>8}")
        print("-" * 90)

        # 定義主要變數顯示順序：const + X1-X8（不含 X9 產業虛擬變數）
        main_var_order = [
            'const',
            'X1',
            'X2_cumulative_turnover_20d',
            'X3_cumulative_return',
            'X4_top10_change_rate',
            'X5_days_since_last',
            'X6_retail_change_rate',
            'X7_log_size',
            'X8_bm_ratio'
        ]
        
        # 變數名稱映射（用於顯示）
        display_names = {
            'const': 'const',
            'X1': 'X1',
            'X2_cumulative_turnover_20d': 'X2_cumulative_turnover_20d',
            'X3_cumulative_return': 'X3_cumulative_return',
            'X4_top10_change_rate': 'X4_top10_change_rate',
            'X5_days_since_last': 'X5_days_since_last',
            'X6_retail_change_rate': 'X6_retail_change_rate',
            'X7_log_size': 'X7_log_size',
            'X8_bm_ratio': 'X8_bm_ratio'
        }
        
        # 顯示主要變數結果
        for var in main_var_order:
            if var in results.params.index:
                coef = results.params[var]
                se = results.bse[var]
                t = results.tvalues[var]
                p = results.pvalues[var]
                sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                display_name = display_names.get(var, var)
                print(f"{display_name:<30} {coef:>12.6f} {se:>12.6f} {t:>10.2f} {p:>10.4f} {sig:>8}")
        
        print(f"\nR-squared: {results.rsquared:.4f}")
        print(f"Adj. R-squared: {results.rsquared_adj:.4f}")
        print(f"F-statistic: {results.fvalue:.2f}")
        print(f"Prob (F-statistic): {results.f_pvalue:.4f}")
        
        # 顯示產業虛擬變數摘要
        if self.industry_dummies_cols:
            print("\n" + "="*80)
            print(f"產業虛擬變數係數 X9（共 {len(self.industry_dummies_cols)} 個）")
            print("="*80)
            print(f"{'變數':<40} {'係數':>12} {'t值':>10} {'P>|t|':>10} {'顯著性':>8}")
            print("-" * 90)
            
            for var in self.industry_dummies_cols:
                if var in results.params.index:
                    coef = results.params[var]
                    t = results.tvalues[var]
                    p = results.pvalues[var]
                    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                    # 將 X8_ 前綴改為 X9_
                    display_name = var.replace('X8_', 'X9_')
                    print(f"{display_name:<40} {coef:>12.6f} {t:>10.2f} {p:>10.4f} {sig:>8}")
        
        print("="*80 + "\n")

        # 儲存結果
        self.regression_results = results

        return self

    def save_results(self, output_dir=None):
        """儲存迴歸結果"""
        if self.regression_results is None:
            print("警告：沒有迴歸結果可儲存（可能因為樣本不足）\n")
            return
        
        output_dir = output_dir or (PROJECT_ROOT / 'results')
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 儲存摘要
        summary_path = output_dir / 'regression_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(str(self.regression_results.summary()))

        print(f"迴歸結果已儲存至: {summary_path}\n")

        # 儲存係數表
        coef_df = pd.DataFrame({
            'Variable': self.regression_results.params.index,
            'Coefficient': self.regression_results.params.values,
            'Std_Error': self.regression_results.bse.values,
            't_statistic': self.regression_results.tvalues.values,
            'p_value': self.regression_results.pvalues.values,
            'CI_lower': self.regression_results.conf_int()[0].values,
            'CI_upper': self.regression_results.conf_int()[1].values
        })

        coef_path = output_dir / 'regression_coefficients.csv'
        coef_df.to_csv(coef_path, index=False)
        print(f"係數表已儲存至: {coef_path}\n")

        # 印出模型診斷
        print("=== 模型診斷 ===")
        print(f"R-squared: {self.regression_results.rsquared:.4f}")
        print(f"Adj. R-squared: {self.regression_results.rsquared_adj:.4f}")
        print(f"F-statistic: {self.regression_results.fvalue:.4f}")
        print(f"Prob (F-statistic): {self.regression_results.f_pvalue:.4e}")
        print(f"AIC: {self.regression_results.aic:.2f}")
        print(f"BIC: {self.regression_results.bic:.2f}")

        return self


def main():
    """主程式"""
    analyzer = UnifiedRegressionAnalyzer()

    try:
        analyzer.load_and_merge_data()
        analyzer.run_regression()
        analyzer.save_results()

        print("\n階段8完成！")

    except FileNotFoundError as e:
        print(f"\n錯誤：找不到必要的資料檔案")
        print(f"詳細錯誤: {str(e)}")
        print("\n請先執行階段1-7以產生所有必要資料")

    except Exception as e:
        print(f"\n錯誤：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
