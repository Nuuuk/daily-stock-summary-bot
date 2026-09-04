import os
import sys
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)
from data_collector import assemble_full_market_payload

# 初始化 DeepSeek 客户端
def get_deepseek_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def load_environment_config():
    raw_config = os.environ.get("APP_CONFIG_JSON")
    if not raw_config:
        print("[Error] 缺失必要的环境变量: APP_CONFIG_JSON")
        sys.exit(1)

    try:
        config = json.loads(raw_config)
        if not isinstance(config, dict):
            print("[Error] APP_CONFIG_JSON 必须是 JSON object")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[Error] APP_CONFIG_JSON JSON 格式解析失败: {e}")
        sys.exit(1)

    # 切换为获取 DEEPSEEK_API_KEY
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL") or sender_email

    if not all([deepseek_api_key, finnhub_key, sender_email, sender_password]):
        print("[Error] 缺失必要的 API Key 或邮件 SMTP 凭证。")
        sys.exit(1)

    return {
        "config": config,
        "deepseek_api_key": deepseek_api_key,
        "finnhub_key": finnhub_key,
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email
    }

def determine_session_mode():
    """优先读取命令行指定参数，确保排队延误时不发生模式篡改"""
    if len(sys.argv) > 1 and sys.argv[1] in ["pre_market", "mid_day"]:
        return sys.argv[1]

    # 本地或未传参时的兜底：按纽约美东本地时间 12:00 为分界线
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)
    return "pre_market" if now_ny.hour < 12 else "mid_day"

def compress_payload_for_llm(market_payload: dict) -> dict:
    """
    将 Tax Lot 与 Ticker-Level Market Data 分离，
    避免 technical/news 在每个 lot 中重复存储。
    """

    cleaned_lots = []

    for lot in market_payload.get("positions_tax_lots", []):
        cleaned_lots.append({
            "sector": lot.get("sector"),
            "ticker": lot.get("ticker"),
            "broker": lot.get("broker"),
            "buy_date": lot.get("buy_date"),
            "qty": lot.get("quantity"),
            "cost": lot.get("cost_basis"),
            "price": lot.get("current_price"),
            "pnl": lot.get("unrealized_pnl"),
            "pnl_pct": lot.get("unrealized_pnl_pct"),
            "days": lot.get("holding_days"),
            "tax_label": lot.get("tax_status_label"),
            "data_quality": lot.get("data_quality")
        })

    ticker_info = {}

    for ticker, data in market_payload.get("ticker_market_data", {}).items():
        ticker_info[ticker] = {
            "technical": data.get("technical", {}),
            "news": data.get("news", [])[:2]
        }

    return {
        "timestamp": market_payload.get("timestamp"),
        "macro": market_payload.get("macro_environment"),
        "tax_lots": cleaned_lots,
        "portfolio_tickers_data": ticker_info,
        "candidates": market_payload.get(
            "complementary_candidates_market_data",
            []
        )
    }

def generate_llm_analysis_report(client: OpenAI, market_payload: dict, financial_profile: dict, session_mode: str) -> str:
    """调用 DeepSeek 生成策略简报，内置脱敏与重试机制"""
    
    available_cash = financial_profile.get("available_cash", 0.0)
    risk_tolerance = financial_profile.get("risk_tolerance", "Aggressive")
    primary_goal = financial_profile.get("primary_goal", "Capital Appreciation (>1 Year Hold)")
    
    tax_profile = financial_profile.get("tax_profile", {})
    filing_status = tax_profile.get("filing_status", "N/A")
    tax_state = tax_profile.get("tax_state", "N/A")
    estimated_income = tax_profile.get("estimated_annual_income", 0.0)
    ytd_realized_pnl = tax_profile.get("current_ytd_realized_pnl", 0.0)
    loss_carryover = tax_profile.get("prior_year_loss_carryover", 0.0)
    tlh_enabled = tax_profile.get("tax_loss_harvesting_enabled", True)

    session_title = "盘前操作基调与持仓备忘" if session_mode == "pre_market" else "盘中异动与趋势确认扫描"

    compact_payload = compress_payload_for_llm(market_payload)

    system_instruction = """
你是一名服务于进取型中长线投资者的买方对冲基金首席宏观与量化税务配置专家。
请严格恪守以下投资与税务纪律（宪法级指令）：

1. **默认持有立场**：重仓标的持仓 >1 年，绝不追高杀跌。日常波动建议【按兵不动】。
2. **Tax Lots 批次级避税指令（铁律）**：
   - 严禁逐行打印所有持仓批次的冗长表格！**只展示存在税务行动点的特殊批次**（持有 300~364 天且浮盈，或浮亏严重具备 TLH 冲销价值的批次）。
   - 针对临近 365 天的浮盈批次，坚决禁止止盈，警示短期资本利得税惩罚。
   - 针对浮亏批次评估 Tax-Loss Harvesting 时，必须提示 30 天 Wash Sale 规则。
   - 必须结合 3.8% NIIT 附加税及州税摩擦在后台测算，但不在正文中展示计算过程。
3. **关键击球点资金管理**：核心标的发生深度回调（52周低位或 200-DMA 支撑）或出现不可错失催化剂时，动用 5%~10% 现金分批建仓。
4. **持仓赛道诊断与新股互补**：评估算力/半导体、大科技、中概互联等赛道集中风险。从候选池选 1~2 只互补标的，**严禁凭空编造价格**，建仓区间必须基于 current_price、ma_50、ma_200 真实计算（如折价 5%~10%）。
5. **【关键隐私脱敏铁律】**：**严禁在邮件正文中明文显示【预估家庭年收入】与【YTD 已实现资本盈亏】的具体数字**！仅允许显示报税身份（如 Married Filing Jointly）与州（NJ）。
6. **【手机端配色红线】**：必须采用极简浅色卡片风（白/浅灰背景），**严禁深色/黑色背景**。**严禁生成 `<style>` 标签或引入 Tailwind**，所有样式必须使用标签内联样式（Inline CSS）。涨跌标签文字与底色必须高对比度。
7. **输出格式**：直接输出原生 HTML，不要带 ```html 标记，全文控制在 3000 tokens 以内，只给结论，禁止解释推理过程。
"""

    prompt = f"""
请根据以下实时数据，生成一份【{session_title}】HTML 邮件分析。

【用户财务参数（仅供后台计算，切勿在邮件正文展示收入与 YTD 具体金额）】
- 可用现金储备: ${available_cash:,.2f}
- 投资风格与目标: {risk_tolerance} ({primary_goal})
- 报税身份与州: {filing_status} / {tax_state}
- 预估家庭年收入: ${estimated_income:,.2f} (仅测算税率档位)
- YTD 已实现资本盈亏: ${ytd_realized_pnl:,.2f} (仅测算 TLH 冲销)
- 往年结转亏损: ${loss_carryover:,.2f}
- 亏损抵税启用: {tlh_enabled}
- 执行模式: {session_mode}

【市场与持仓 Tax Lots 实时 Payload】
{json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))}

【需要生成的章节内容】（严格按此顺序输出结论）：
1. 宏观与流动性环境（标普/纳指/罗素分化，结合 ^TNX 与 ^VIX 定性）。
2. Tax Lots 税务行动点（只列特殊批次，带表格）。
3. 当日操作决议（明确写【按兵不动】或【分批加仓+具体金额区间】）。
4. 持仓赛道集中度与新股候选（从 complementary_candidates_market_data 中选 1~2 只，附带真实计算的回调区间）。
5. 基本面新闻风险提炼（过滤短期噪音）。
6. 脱敏确认声明。
"""

    max_retries = 3
    delay = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Info] 正在调用 DeepSeek 生成简报 (第 {attempt} 次)...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
                stream=False
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("DeepSeek 输出被截断")
                
            raw_text = response.choices[0].message.content or ""
            cleaned_html = raw_text.replace("```html", "").replace("```", "").strip()
            
            if len(cleaned_html) < 200:
                raise ValueError(f"返回内容过短 (仅 {len(cleaned_html)} 字符)")
                
            cleaned_html_lower = cleaned_html.lower()
            if "<style" in cleaned_html_lower or "<script" in cleaned_html_lower:
                raise ValueError("模型输出包含禁止的 <style> 或 <script> 标签")

            print(f"[Success] 简报生成成功！输出长度: {len(cleaned_html)} 字符")
            return cleaned_html

        except (APIConnectionError, APITimeoutError, RateLimitError, ValueError) as e:
            print(f"[Warning] 触发重试 ({type(e).__name__}): {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(f"DeepSeek API 重试耗尽: {e}")
                
        except APIStatusError as e:
            print(f"[Warning] DeepSeek API 状态报错 ({e.status_code}): {e}")
            if e.status_code >= 500 and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError("DeepSeek API 致命错误，终止运行。")

def send_email_notification(html_content: str, subject_prefix: str, config_env: dict):
    sender_email = config_env["sender_email"]
    sender_password = config_env["sender_password"]
    receiver_email = config_env["receiver_email"]

# 强制获取美东纽约时区时间（自动适应 EDT 夏令时 / EST 冬令时）
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)
    now_str = now_ny.strftime('%Y-%m-%d %H:%M %Z')
    
    subject = f"[{subject_prefix}] 美股持仓与决策简报 ({now_str})"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Stock Summary Bot <{sender_email}>"
    msg["To"] = receiver_email

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"[Success] 简报邮件已成功发送至: {receiver_email}")
    except Exception as e:
        print(f"[Error] 邮件发送失败: {e}")
        sys.exit(1)


def main():
    print(f"[{datetime.now()}] 启动 Daily Stock Positions Summary Bot...")
    
    env_config = load_environment_config()
    app_config = env_config["config"]
    
    positions = app_config.get("portfolio", [])
    financial_profile = app_config.get("financial_profile", {})
    session_mode = determine_session_mode()

    print(f"[Info] 运行模式: {session_mode}, 监控 Tax Lots 批次总数: {len(positions)}")

    print("[Info] 正在拉取宏观指数、个股技术面与 Finnhub 新闻...")
    market_payload = assemble_full_market_payload(
        positions=positions,
        finnhub_api_key=env_config["finnhub_key"],
        days_back=1 if session_mode == "pre_market" else 0
    )

    print("[Info] 正在调用 DeepSeek API 生成策略简报...")
    deepseek_key = env_config["deepseek_api_key"]
    client = get_deepseek_client(deepseek_key)
    
    html_report = generate_llm_analysis_report(
        client=client,
        market_payload=market_payload,
        financial_profile=financial_profile,
        session_mode=session_mode
    )

    prefix = "盘前决策" if session_mode == "pre_market" else "盘中扫描"
    send_email_notification(html_report, prefix, env_config)
    print("[Done] 全流程执行完毕。")


if __name__ == "__main__":
    main()
