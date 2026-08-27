import os
import sys
import json
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google import genai
from google.genai import types

from data_collector import assemble_full_market_payload


def load_environment_config():
    raw_config = os.environ.get("APP_CONFIG_JSON")
    if not raw_config:
        print("[Error] 缺失必要的环境变量: APP_CONFIG_JSON")
        sys.exit(1)

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as e:
        print(f"[Error] APP_CONFIG_JSON JSON 格式解析失败: {e}")
        sys.exit(1)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL", sender_email)

    if not all([gemini_key, finnhub_key, sender_email, sender_password]):
        print("[Error] 缺失必要的 API Key 或邮件 SMTP 凭证。")
        sys.exit(1)

    return {
        "config": config,
        "gemini_key": gemini_key,
        "finnhub_key": finnhub_key,
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email
    }


def determine_session_mode():
    if len(sys.argv) > 1 and sys.argv[1] in ["pre_market", "mid_day"]:
        return sys.argv[1]

    current_hour_utc = datetime.utcnow().hour
    if 11 <= current_hour_utc <= 14:
        return "pre_market"
    return "mid_day"


def generate_llm_analysis_report(gemini_client: genai.Client, market_payload: dict, financial_profile: dict, session_mode: str) -> str:
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

    system_instruction = """
你是一名服务于进取型长线投资者的买方对冲基金首席宏观与量化税务配置专家。
请严格恪守以下投资与税务纪律：
1. **默认持有立场（Default Hold）**：投资者追求极低换手率（持仓 >1 年），绝不追高，避免杀跌。日常行情波动必须保持【按兵不动】。
2. **Tax Lots 批次级避税指令**：
   - 当特定批次持有接近 365 天（300~364天）且处于浮盈时，坚决禁止止盈建议，警示短期资本利得税惩罚；
   - 若某特定批次浮亏严重且该投资者 YTD 已实现盈利较大，评估 Tax-Loss Harvesting 冲销价值（注意提示 30 天 Wash Sale 规则）。
   - 考虑报税身份、收入门槛（如 3.8% NIIT 附加税）及所在州税摩擦。
3. **关键击球点（Fat Pitch）资金管理**：
   - 基于传入的【可用现金储备】，绝不轻易建议开仓。仅在核心标的发生非理性深度回调（进入 52 周低位或 200-DMA 关键支撑）或出现不可错失的催化剂时，建议动用 5%~10% 现金分批建仓。
4. **输出格式**：直接输出现代、极简、高对比度、带内联 CSS 样式的原生 HTML 正文，禁止包含 Markdown 代码块标记（如 ```html）。
"""

    prompt = f"""
请分析以下实时市场 Payload 与个人财务参数，生成一份【{session_title}】HTML 邮件。

【用户财务与税务配置（动态安全变量）】
- 可用现金储备: ${available_cash:,.2f}
- 投资风格与目标: {risk_tolerance} ({primary_goal})
- 报税身份与州: {filing_status} / {tax_state}
- 预估家庭年收入: ${estimated_income:,.2f}
- YTD 已实现资本盈亏 (Realized P&L): ${ytd_realized_pnl:,.2f}
- 往年结转资本亏损 (Loss Carryover): ${loss_carryover:,.2f}
- 亏损抵税策略已启用: {tlh_enabled}
- 当前执行模式: {session_mode}

【市场与各批次 Tax Lots 实时 Payload】
{json.dumps(market_payload, indent=2, ensure_ascii=False)}

【HTML 邮件排版规范】
1. **宏观与流动性环境**：标普/纳指/罗素分化，结合美债10年期收益率(^TNX)与恐慌指数(^VIX)给出大盘定性。
2. **Tax Lots 持仓明细与税收时钟表 (表格呈现)**：
   - 列出：券商(Broker)、代码(Ticker)、买入日期、持有天数、成本价、现价、浮盈亏($/%)、税收状态标签。
   - 对临近 1 年的长税冲刺批次以醒目黄色/橙色突出显示。
3. **当日操作决议（克制、果断）**：
   - 明确给出【维持现状/按兵不动】、【分批加仓】或【税收亏损收割】。若加仓，需换算基于可用现金的具体金额区间。
4. **基本面与新闻风险提炼**：简要概括影响长期逻辑的新闻要点，过滤短期噪音。
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
    )

    cleaned_html = response.text.replace("```html", "").replace("```", "").strip()
    return cleaned_html


def send_email_notification(html_content: str, subject_prefix: str, config_env: dict):
    sender_email = config_env["sender_email"]
    sender_password = config_env["sender_password"]
    receiver_email = config_env["receiver_email"]

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    subject = f"[{subject_prefix}] 美股持仓与决策简报 ({now_str} EST)"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Stock Summary Bot <{sender_email}>"
    msg["To"] = receiver_email

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
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

    print("[Info] 正在调用 Gemini 3.7 Flash 生成策略简报...")
    gemini_client = genai.Client(api_key=env_config["gemini_key"])
    
    html_report = generate_llm_analysis_report(
        gemini_client=gemini_client,
        market_payload=market_payload,
        financial_profile=financial_profile,
        session_mode=session_mode
    )

    prefix = "盘前决策" if session_mode == "pre_market" else "盘中扫描"
    send_email_notification(html_report, prefix, env_config)
    print("[Done] 全流程执行完毕。")


if __name__ == "__main__":
    main()
