#!/usr/bin/env python3
"""
TEJ 工具輔助函式
提供統一的介面呼叫各種 TEJ 資料工具
"""
import subprocess
import pandas as pd
import os
from pathlib import Path
from datetime import datetime


class TEJDataFetcher:
    """TEJ 資料擷取器"""

    def __init__(self, base_dir='.'):
        """
        初始化擷取器

        Parameters:
        -----------
        base_dir : str
            TEJ 工具所在目錄
        """
        self.base_dir = base_dir
        self.tools = {
            'ABETAD1': f'{base_dir}/tej_tool_TWN_ABETAD1.py',
            'ATINST1W': f'{base_dir}/tej_tool_TWN_ATINST1W.py',
            'ADCSHR': f'{base_dir}/tej_tool_TWN_ADCSHR.py',
            'AAPRCW1': f'{base_dir}/tej_tool_TWN_AAPRCW1.py',
            'AGIN': f'{base_dir}/tej_tool_TWN_AGIN.py'
        }

    def fetch_beta_data(self, coid, start_date, end_date):
        """
        取得 Beta 與報酬率資料

        Parameters:
        -----------
        coid : str
            股票代號
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)

        Returns:
        --------
        pd.DataFrame
        """
        return self._fetch_data(
            'ABETAD1',
            coid, start_date, end_date,
            output_dir='output_abetad1'
        )

    def fetch_institutional_data(self, coid, start_date, end_date):
        """
        取得三大法人買賣超資料

        Parameters:
        -----------
        coid : str
            股票代號
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)

        Returns:
        --------
        pd.DataFrame
        """
        return self._fetch_data(
            'ATINST1W',
            coid, start_date, end_date,
            output_dir='output_atinst1w'
        )

    def fetch_shareholding_data(self, coid, start_date, end_date):
        """
        取得股權分散表（董監事持股）資料

        Parameters:
        -----------
        coid : str
            股票代號
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)

        Returns:
        --------
        pd.DataFrame
        """
        return self._fetch_data(
            'ADCSHR',
            coid, start_date, end_date,
            output_dir='output_adcshr'
        )

    def fetch_trading_data(self, coid, start_date, end_date):
        """
        取得週成交資料

        Parameters:
        -----------
        coid : str
            股票代號
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)

        Returns:
        --------
        pd.DataFrame
        """
        return self._fetch_data(
            'AAPRCW1',
            coid, start_date, end_date,
            output_dir='output_aaprcw1'
        )

    def fetch_margin_data(self, coid, start_date, end_date):
        """
        取得融資融券餘額資料

        Parameters:
        -----------
        coid : str
            股票代號
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)

        Returns:
        --------
        pd.DataFrame
        """
        return self._fetch_data(
            'AGIN',
            coid, start_date, end_date,
            output_dir='output_agin'
        )

    def _fetch_data(self, tool_name, coid, start_date, end_date, output_dir):
        """
        通用資料擷取方法

        Parameters:
        -----------
        tool_name : str
            工具名稱
        coid : str
            股票代號
        start_date : str
            開始日期
        end_date : str
            結束日期
        output_dir : str
            輸出目錄

        Returns:
        --------
        pd.DataFrame
        """
        tool_path = self.tools.get(tool_name)
        if not tool_path or not os.path.exists(tool_path):
            print(f"警告：找不到工具 {tool_name}")
            return None

        try:
            # 執行 TEJ 工具
            cmd = [
                'python', tool_path,
                '-c', str(coid),
                '-start', start_date,
                '-end', end_date,
                '-o', output_dir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return None

            # 找尋輸出檔案
            output_files = list(Path(output_dir).glob(f'*_{coid}_*.csv'))
            if not output_files:
                return None

            # 讀取最新的檔案
            latest_file = max(output_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)

            return df

        except Exception as e:
            print(f"錯誤：擷取 {tool_name} 資料時發生錯誤: {str(e)}")
            return None


def format_date(date, format_str='%Y-%m-%d'):
    """
    格式化日期

    Parameters:
    -----------
    date : datetime or str
        日期
    format_str : str
        格式字串

    Returns:
    --------
    str
    """
    if isinstance(date, str):
        return date
    return date.strftime(format_str)
