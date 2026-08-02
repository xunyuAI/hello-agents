# -*- coding: utf-8 -*-
# 本文件由 autogen_software_team.py 协作结果提取生成。
# 运行方式: streamlit run generated_bitcoin_app.py

import streamlit as st
import requests
from datetime import datetime

# 页面配置
st.set_page_config(page_title="比特币实时价格", page_icon="₿", layout="centered")

# ---- 数据获取函数 ----
@st.cache_data(ttl=60, show_spinner=False)  # 缓存60秒，避免手动刷新时频繁请求
def fetch_btc_price():
    """
    调用 CoinGecko API 获取比特币价格信息
    返回: dict 包含 current_price, price_change_24h, price_change_percentage_24h
    失败时抛出异常
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": "bitcoin",
        "order": "market_cap_desc",
        "sparkline": "false",
        "price_change_percentage": "24h"
    }
    headers = {"accept": "application/json"}

    resp = requests.get(url, params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        resp.raise_for_status()  # 触发 HTTPError

    data = resp.json()
    if not data:
        raise ValueError("API 返回空数据")

    coin = data[0]
    return {
        "price": coin["current_price"],
        "change_24h": coin.get("price_change_24h", 0),
        "change_percent_24h": coin.get("price_change_percentage_24h", 0),
        "last_updated": datetime.now().strftime("%H:%M:%S")
    }

# ---- UI 构建 ----
st.title("₿ 比特币价格追踪器")
st.caption("实时价格与24小时趋势 · 数据来源 CoinGecko")

# 刷新按钮状态
if "refresh_pressed" not in st.session_state:
    st.session_state.refresh_pressed = False

def refresh_callback():
    st.session_state.refresh_pressed = True
    st.cache_data.clear()  # 清除缓存，强制重新获取

col1, col2 = st.columns([3, 1])
with col2:
    st.button("🔄 刷新价格", on_click=refresh_callback, use_container_width=True)

# 主要展示区域
price_placeholder = st.empty()
error_placeholder = st.empty()

# 获取数据（显示 spinner）
try:
    with st.spinner("正在获取最新价格..."):
        price_data = fetch_btc_price()

        # 成功展示
        price = price_data["price"]
        change_24h = price_data["change_24h"]
        change_percent_24h = price_data["change_percent_24h"]
        last_update = price_data["last_updated"]

        # 价格卡片（使用 markdown 大字体）
        price_html = f"""
        <div style="text-align: center; padding: 30px 0;">
            <div style="font-size: 3.5em; font-weight: bold; color: #FF9900;">${price:,.2f}</div>
            <div style="font-size: 1.2em; margin-top: 8px;">
                <span style="color: {'green' if change_24h >= 0 else 'red'}; font-weight: bold;">
                    {'+' if change_24h >=0 else ''}{change_24h:,.2f} USD
                    ({'+' if change_percent_24h >=0 else ''}{change_percent_24h:.2f}%)
                </span>
            </div>
            <div style="margin-top: 15px; color: #888; font-size: 0.9em;">
                最后更新: {last_update}
            </div>
        </div>
        """
        price_placeholder.markdown(price_html, unsafe_allow_html=True)

        # 清除可能存在的错误信息
        error_placeholder.empty()

except requests.exceptions.RequestException as e:
    error_placeholder.error(f"⚠️ 网络请求失败: {str(e)}")
    price_placeholder.markdown(
        '<div style="text-align:center; padding:20px; color:#888;">无法获取实时价格，请点击刷新按钮重试</div>',
        unsafe_allow_html=True
    )
except Exception as e:
    error_placeholder.error(f"⚠️ 数据解析异常: {str(e)}")
    price_placeholder.markdown(
        '<div style="text-align:center; padding:20px; color:#888;">数据异常，请稍后重试</div>',
        unsafe_allow_html=True
    )

# 重置刷新状态
if st.session_state.refresh_pressed:
    st.session_state.refresh_pressed = False

# 页脚说明
st.markdown("---")
st.caption("提示：数据来自公共 API，更新频率受限于 CoinGecko 调用限制，请勿频繁刷新。")
