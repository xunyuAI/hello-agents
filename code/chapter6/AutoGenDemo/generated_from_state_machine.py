import streamlit as st
import requests
import time

# ---------- 初始化 session_state ----------
if 'last_price' not in st.session_state:
    st.session_state.last_price = None
    st.session_state.last_change = None
    st.session_state.last_error = None
    st.session_state.last_request_time = 0
    st.session_state.auto_refresh_enabled = False
    st.session_state.initial_fetch_done = False

# ---------- 自动刷新：通过 meta 标签实现每 60 秒刷新整个页面 ----------
if st.session_state.auto_refresh_enabled:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True
    )

# ---------- 数据获取函数 ----------
def fetch_data():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.last_price = data['bitcoin']['usd']
        st.session_state.last_change = data['bitcoin']['usd_24h_change']
        st.session_state.last_error = None
    except Exception as e:
        # 用户看到的错误信息简洁友好，不暴露内部细节
        st.session_state.last_error = "无法获取价格，请稍后重试。"
        # 控制台输出详细错误，便于调试
        print(f"[ERROR] 获取数据失败: {e}")
    finally:
        st.session_state.last_request_time = time.time()

# ---------- 首次自动获取 ----------
if not st.session_state.initial_fetch_done:
    fetch_data()
    st.session_state.initial_fetch_done = True

# ---------- 若开启自动刷新，且距离上次请求已超过 60 秒，则立即拉取新数据 ----------
if st.session_state.auto_refresh_enabled:
    if time.time() - st.session_state.last_request_time >= 60:
        fetch_data()
        st.rerun()

# ---------- UI 布局 ----------
st.title("比特币价格监控")
st.caption("数据来源：CoinGecko | 开发者示例")

col1, col2 = st.columns(2)

with col1:
    price_str = f"${st.session_state.last_price:,.2f}" if st.session_state.last_price is not None else "---"
    st.metric("当前价格 (USD)", price_str)

with col2:
    if st.session_state.last_change is not None:
        change = st.session_state.last_change
        # 零或接近零（绝对值小于 0.005%）显示灰色，否则正绿负红
        if abs(change) < 0.005:
            color = "gray"
        elif change > 0:
            color = "green"
        else:
            color = "red"
        change_str = f"{change:+.2f}%"
        st.markdown(
            f"<h4 style='color:{color};'>{change_str}</h4>",
            unsafe_allow_html=True
        )
        # 添加一个小的 caption 表示这是 24 小时涨跌幅
        st.caption("24小时涨跌幅")
    else:
        st.markdown("---")
        st.caption("24小时涨跌幅")

# 错误提示及过时数据警告
if st.session_state.last_error:
    st.warning(f"⚠️ {st.session_state.last_error}")
    if st.session_state.last_price is not None:
        st.info("当前显示的是上一次成功获取的数据（可能已过时）。")

# 手动刷新按钮
if st.button("刷新价格"):
    with st.spinner("正在获取最新数据..."):
        fetch_data()
    # 按钮点击后 Streamlit 自动重新运行脚本，无需额外操作

# 自动刷新开关
auto = st.checkbox(
    "启用自动刷新（每 60 秒）",
    value=st.session_state.auto_refresh_enabled
)
st.session_state.auto_refresh_enabled = auto

PHASE_DONE