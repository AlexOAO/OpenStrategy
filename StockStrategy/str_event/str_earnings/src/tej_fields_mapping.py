"""
TEJ 工具實際欄位對應表
根據各 TEJ 工具的實際欄位定義建立對應關係
"""

# TWN/ATINST1W - 三大法人買賣超週度
ATINST1W_FIELDS = {
    # 買賣超股數（千股）
    'foreign_shares': 'sqfii_ex1_2',      # 外資本週以來買賣超(千股)
    'trust_shares': 'sfund_ex_2',          # 投信本週以來買賣超(千股)
    'dealer_shares': 'sdlr_ex_2',          # 自營本週以來買賣超(千股)
    'total_shares': 'sttl_ex_2',           # 合計本週以來買賣超(千股)

    # 買賣超金額（千元）
    'foreign_amount': 'sqfii_amt_2',       # 外資本週買賣超金額(千元)
    'trust_amount': 'sfund_amt_2',         # 投信本週買賣超金額(千元)
    'dealer_amount': 'sdlr_amt_2',         # 自營本週買賣超金額(千元)
    'total_amount': 'sttl_amt_2',          # 合計本週買賣超金額(千元)

    # 持股相關
    'foreign_holdings': 'ttl_stk',         # 外資總投資股數(千股)
    'trust_holdings': 'fld008',            # 投信持股數(千股)
    'dealer_holdings': 'fld011',           # 自營持股數(千股)
}

# TWN/ADCSHR - 股權分散表
ADCSHR_FIELDS = {
    'total_holders': 'ttl_man',            # 集保總人數
    'total_shares': 'ttl_noa',             # 集保總張數(千股)

    # 董監事持股需要從大戶持股推算
    # 通常 400 張以上視為董監事和大股東
    # 欄位 f01, g01, h01, i01 代表不同級距的人數
    # 欄位 f02, g02, h02, i02 代表不同級距的千股數
}

# TWN/AGIN - 融資融券餘額
AGIN_FIELDS = {
    # 融資
    'margin_balance_shares': 'gin0',       # 融資餘額(張)
    'margin_balance_amount': 'l0ng_ta',    # 融資餘額(千元)
    'margin_buy': 'buy_l',                 # 融資買進(張)
    'margin_sell': 'sell_l',               # 融資賣出(張)
    'margin_change': 'gin1',               # 融資增減(張)
    'margin_change_ratio': 'gin2',         # 融資增減比率
    'margin_utilization': 'gin3',          # 融資使用率

    # 融券
    'short_balance_shares': 'gin4',        # 融券餘額(張)
    'short_balance_amount': 'short_ta',    # 融券餘額(千元)
    'short_buy': 'buy_s',                  # 融券買進(張)
    'short_sell': 'sell_s',                # 融券賣出(張)
    'short_change': 'gin5',                # 融券增減(張)
}

# TWN/AAPRCW1 - 週成交資料
AAPRCW1_FIELDS = {
    'volume': 'volume',                    # 成交量(千股)_週
    'amount': 'amount',                    # 成交值(千元)_週
    'close': 'close_w',                    # 收盤價(元)_週
    'return': 'roi',                       # 報酬率％_週
    'turnover': 'turnover',                # 週轉率％_週
    'market_value': 'mv',                  # 市值(百萬元)
}

# TWN/ASHR1A - 借券賣出與融券賣出
ASHR1A_FIELDS = {
    # 買賣超（張）
    'foreign_net': 'qfii_ex',              # 外資買賣超(張)
    'trust_net': 'fund_ex',                # 投信買賣超(張)
    'dealer_net': 'dlr_ex',                # 自營買賣超(張)
    'total_net': 'ttl_ex',                 # 合計買賣超(張)
    'all_net': 'all_ex',                   # 一般現股買賣超(張)
    'margin_net': 'gin_ex',                # 信用交易買賣超(張)
}


def get_institutional_buying(df):
    """
    計算三大法人淨買超（金額）

    Parameters:
    -----------
    df : pd.DataFrame
        ATINST1W 資料

    Returns:
    --------
    float : 淨買超金額總和（千元）
    """
    foreign = df[ATINST1W_FIELDS['foreign_amount']].fillna(0).sum()
    trust = df[ATINST1W_FIELDS['trust_amount']].fillna(0).sum()
    dealer = df[ATINST1W_FIELDS['dealer_amount']].fillna(0).sum()

    return foreign + trust + dealer


def get_margin_balance(df):
    """
    取得融資餘額

    Parameters:
    -----------
    df : pd.DataFrame
        AGIN 資料

    Returns:
    --------
    Series : 融資餘額（張）
    """
    return df[AGIN_FIELDS['margin_balance_shares']]


def get_short_balance(df):
    """
    取得融券餘額

    Parameters:
    -----------
    df : pd.DataFrame
        AGIN 資料

    Returns:
    --------
    Series : 融券餘額（張）
    """
    return df[AGIN_FIELDS['short_balance_shares']]


def estimate_insider_holdings(df):
    """
    估計董監事持股
    ADCSHR 沒有直接的董監事持股欄位
    使用大股東持股作為代理變數（400張以上）

    Parameters:
    -----------
    df : pd.DataFrame
        ADCSHR 資料

    Returns:
    --------
    float : 估計的大股東持股（千股）
    """
    # 檢查是否有 f02, g02, h02, i02 等大額持股欄位
    # 這些通常代表 400-600張, 600-800張, 800-1000張, 1000張以上
    large_holder_fields = ['f02', 'g02', 'h02', 'i02']

    total = 0
    for field in large_holder_fields:
        if field in df.columns:
            total += df[field].fillna(0).sum()

    return total


def get_volume(df):
    """
    取得成交量

    Parameters:
    -----------
    df : pd.DataFrame
        AAPRCW1 資料

    Returns:
    --------
    Series : 成交量（千股）
    """
    return df[AAPRCW1_FIELDS['volume']]
