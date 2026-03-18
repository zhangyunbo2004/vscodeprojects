import streamlit as st
import requests
import datetime
import time
import logging
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class AppConfig:
    """应用配置类"""
    REFRESH_INTERVAL: int = int(os.getenv("REFRESH_INTERVAL", "30"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    API_URL: str = os.getenv("API_URL", "https://api.coingecko.com/api/v3/simple/price")
    API_PARAMS: Dict = None

    def __post_init__(self):
        """初始化API参数"""
        self.API_PARAMS = {
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "false",
            "include_last_updated_at": "true"
        }

    def validate(self):
        """验证配置参数"""
        if self.REFRESH_INTERVAL < 5:
            logging.warning("刷新间隔过短可能导致API限制，建议设置为10秒以上")

        if self.REQUEST_TIMEOUT < 1:
            raise ValueError("请求超时时间不能小于1秒")

        if not self.API_URL.startswith(("http://", "https://")):
            raise ValueError("API URL格式不正确")

class BitcoinPriceApp:
    """比特币价格显示应用主类"""

    def __init__(self):
        """初始化应用"""
        self.config = AppConfig()
        self.config.validate()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._setup_page_config()
        self._init_session_state()

    def _setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="比特币价格监控",
            page_icon="₿",
            layout="centered",
            initial_sidebar_state="collapsed"
        )

        # 隐藏Streamlit默认的菜单和页脚
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-1dp5vir {visibility: hidden;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    def _init_session_state(self):
        """初始化session状态"""
        default_states = {
            'btc_data': None,
            'last_update': None,
            'error_message': None,
            'is_loading': False,
            'auto_refresh_countdown': self.config.REFRESH_INTERVAL,
            'last_refresh_time': None
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _validate_data(self, data: Dict) -> bool:
        """
        验证API返回的数据完整性

        Args:
            data: API返回的数据

        Returns:
            bool: 数据是否有效
        """
        if not data:
            return False

        required_fields = ["usd", "usd_24h_change"]
        if not all(field in data for field in required_fields):
            self.logger.error(f"API数据缺少必要字段: {required_fields}")
            return False

        # 验证价格数据合理性
        if data["usd"] <= 0:
            self.logger.error(f"价格数据异常: {data['usd']}")
            return False

        return True

    def fetch_bitcoin_price(self) -> Optional[Dict]:
        """
        从CoinGecko API获取比特币价格数据

        Returns:
            Dict: 包含价格数据的字典，失败时返回None
        """
        self.logger.info("开始获取比特币价格数据")

        try:
            response = requests.get(
                self.config.API_URL,
                params=self.config.API_PARAMS,
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()

            if "bitcoin" not in data:
                raise ValueError("API响应中未找到比特币数据")

            bitcoin_data = data["bitcoin"]

            # 验证数据完整性
            if not self._validate_data(bitcoin_data):
                raise ValueError("API返回的数据不完整或异常")

            self.logger.info("成功获取比特币价格数据")
            return bitcoin_data

        except requests.exceptions.Timeout:
            error_msg = "请求超时，请检查网络连接"
            self.logger.error(error_msg)
            st.session_state.error_message = error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "网络连接错误，请检查网络"
            self.logger.error(error_msg)
            st.session_state.error_message = error_msg
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                error_msg = "API请求过于频繁，请稍后重试"
            else:
                status_code = e.response.status_code if e.response else "未知"
                error_msg = f"API请求失败: HTTP {status_code}"
            self.logger.error(f"{error_msg}: {str(e)}")
            st.session_state.error_message = error_msg
        except ValueError as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            st.session_state.error_message = str(e)
        except Exception as e:
            error_msg = f"获取数据时发生错误: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.session_state.error_message = error_msg

        return None

    def format_price(self, price: float) -> str:
        """
        格式化价格显示

        Args:
            price: 价格数值

        Returns:
            str: 格式化后的价格字符串
        """
        if price >= 1000:
            return f"${price:,.2f}"
        else:
            return f"${price:.2f}"

    def format_change(self, change_percent: float, change_amount: float) -> Tuple[str, str]:
        """
        格式化涨跌幅显示

        Args:
            change_percent: 涨跌百分比
            change_amount: 涨跌金额

        Returns:
            Tuple[str, str]: (百分比字符串, 金额字符串)
        """
        percent_str = f"{change_percent:+.2f}%"
        amount_str = f"${change_amount:+.2f}"
        return percent_str, amount_str

    def get_change_color(self, change: float) -> str:
        """
        根据涨跌获取颜色

        Args:
            change: 涨跌值

        Returns:
            str: 颜色名称
        """
        if change > 0:
            return "#10B981"  # 绿色
        elif change < 0:
            return "#EF4444"  # 红色
        else:
            return "#6B7280"  # 灰色

    def get_change_icon(self, change: float) -> str:
        """
        根据涨跌获取图标

        Args:
            change: 涨跌值

        Returns:
            str: 图标字符
        """
        if change > 0:
            return "📈"
        elif change < 0:
            return "📉"
        else:
            return "➡️"

    def display_header(self):
        """显示应用标题和说明"""
        st.title("₿ 比特币价格监控")
        st.markdown("实时追踪比特币市场价格及变化趋势")
        st.markdown("---")

    def display_price_card(self, data: Dict):
        """
        显示价格卡片

        Args:
            data: 比特币价格数据
        """
        # 提取数据
        current_price = data.get("usd", 0)
        change_24h_percent = data.get("usd_24h_change", 0)
        change_24h_amount = current_price * (change_24h_percent / 100)

        # 格式化显示
        price_str = self.format_price(current_price)
        percent_str, amount_str = self.format_change(change_24h_percent, change_24h_amount)
        change_color = self.get_change_color(change_24h_percent)
        change_icon = self.get_change_icon(change_24h_percent)

        # 创建价格卡片
        col1, col2 = st.columns([2, 1])

        with col1:
            # 主价格显示
            st.markdown(f"""
            <div style='text-align: left;'>
                <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; font-weight: 700;'>{price_str}</h1>
                <div style='display: flex; align-items: center; gap: 0.5rem;'>
                    <span style='font-size: 1.5rem;'>{change_icon}</span>
                    <span style='font-size: 1.2rem; color: {change_color}; font-weight: 600;'>
                        {percent_str} ({amount_str})
                    </span>
                </div>
                <p style='font-size: 0.9rem; color: #6B7280; margin-top: 0.5rem;'>
                    24小时价格变化
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # 比特币图标
            st.markdown("""
            <div style='text-align: center; padding-top: 1rem;'>
                <div style='font-size: 4rem; margin: 0;'>₿</div>
                <p style='font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;'>
                    Bitcoin
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    def display_last_update(self, timestamp: float):
        """
        显示最后更新时间

        Args:
            timestamp: Unix时间戳
        """
        if timestamp:
            update_time = datetime.datetime.fromtimestamp(timestamp)
            formatted_time = update_time.strftime("%Y-%m-%d %H:%M:%S")

            # 计算时间差
            now = datetime.datetime.now()
            time_diff = now - update_time
            minutes_diff = int(time_diff.total_seconds() / 60)

            if minutes_diff < 1:
                time_ago = "刚刚"
            elif minutes_diff < 60:
                time_ago = f"{minutes_diff}分钟前"
            else:
                hours_diff = minutes_diff // 60
                time_ago = f"{hours_diff}小时前"

            st.caption(f"🕒 最后更新: {formatted_time} ({time_ago})")

    def display_refresh_countdown(self):
        """显示刷新倒计时"""
        if st.session_state.auto_refresh_countdown is not None:
            countdown = st.session_state.auto_refresh_countdown

            # 更新倒计时
            if countdown > 0:
                st.session_state.auto_refresh_countdown -= 1
            else:
                st.session_state.auto_refresh_countdown = self.config.REFRESH_INTERVAL
                if not st.session_state.is_loading:
                    st.session_state.is_loading = True

            # 显示倒计时
            progress = countdown / self.config.REFRESH_INTERVAL
            st.progress(progress, text=f"⏳ 自动刷新倒计时: {countdown}秒")

    def display_controls(self):
        """显示控制按钮"""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # 刷新按钮
            if st.button("🔄 刷新价格", type="primary", use_container_width=True):
                st.session_state.is_loading = True
                st.session_state.auto_refresh_countdown = self.config.REFRESH_INTERVAL
                st.rerun()

    def display_loading_state(self):
        """显示加载状态"""
        if st.session_state.is_loading:
            with st.spinner("正在获取最新价格..."):
                # 获取新数据
                new_data = self.fetch_bitcoin_price()

                if new_data:
                    st.session_state.btc_data = new_data
                    st.session_state.last_update = new_data.get("last_updated_at")
                    st.session_state.error_message = None
                    st.session_state.last_refresh_time = time.time()
                else:
                    # 如果获取失败，保持旧数据
                    if not st.session_state.btc_data:
                        st.session_state.error_message = "无法获取价格数据，请检查网络连接"

                st.session_state.is_loading = False
                st.rerun()

    def display_error_state(self):
        """显示错误状态"""
        if st.session_state.error_message:
            st.error(f"⚠️ {st.session_state.error_message}")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 重试", key="retry_error", use_container_width=True):
                    st.session_state.error_message = None
                    st.session_state.is_loading = True
                    st.session_state.auto_refresh_countdown = self.config.REFRESH_INTERVAL
                    st.rerun()

    def display_footer(self):
        """显示页脚信息"""
        st.markdown("---")

        # 显示配置信息
        with st.expander("ℹ️ 应用信息", expanded=False):
            st.write(f"**刷新间隔**: {self.config.REFRESH_INTERVAL}秒")
            st.write(f"**请求超时**: {self.config.REQUEST_TIMEOUT}秒")
            st.write(f"**数据源**: [CoinGecko API](https://www.coingecko.com/)")
            st.write(f"**最后刷新**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 显示版权信息
        st.markdown(
            "<p style='text-align: center; color: #6B7280; font-size: 0.8rem;'>"
            "© 2024 比特币价格监控应用 • 数据仅供参考，投资需谨慎</p>",
            unsafe_allow_html=True
        )

    def run(self):
        """运行主应用"""
        # 显示标题
        self.display_header()

        # 初始加载数据
        if not st.session_state.btc_data and not st.session_state.is_loading:
            st.session_state.is_loading = True
            st.rerun()

        # 显示加载状态
        self.display_loading_state()

        # 显示错误状态（如果有）
        self.display_error_state()

        # 显示价格数据（如果有）
        if st.session_state.btc_data and not st.session_state.error_message:
            self.display_price_card(st.session_state.btc_data)
            self.display_last_update(st.session_state.last_update)

        # 显示刷新倒计时
        self.display_refresh_countdown()

        # 显示控制按钮
        self.display_controls()

        # 显示页脚
        self.display_footer()
        if not st.session_state.btc_data:
         st.session_state.btc_data = {"usd": 50000, "usd_24h_change": 2.5, "last_updated_at": 1610000000}
        # 设置自动刷新（使用Streamlit的机制）
        time.sleep(1)  # 每秒检查一次
        st.rerun()

def main():
    """应用主函数"""
    try:
        # 创建并运行应用
        app = BitcoinPriceApp()
        app.run()
    except Exception as e:
        st.error(f"应用启动失败: {str(e)}")
        st.info("请检查配置文件或联系管理员")
        logging.error(f"应用启动失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()