import os
import sys
import json
import time
import datetime
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from openai import OpenAI
from data_collector import assemble_full_market_payload

# 初始化 DeepSeek 客户端
def get_deepseek_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2):
    """
    通用重试装饰器：处理大模型 Free Tier 偶发的 429 (Rate Limit) 与 503 (Overloaded)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    # 捕获常见的限流与过载状态码
                    if any(code in error_msg for code in ["429", "503", "ResourceExhausted", "Overloaded"]) or attempt < max_retries:
                        if attempt == max_retries:
                            print(f"[Fatal] 超过最大重试次数 ({max_retries})，调用失败: {error_msg}")
                            raise e
                        print(f"[Warning] 触发 API 频控或临时过载 ({error_msg})，等待 {delay} 秒后进行第 {attempt + 1} 次重试...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise e
        return wrapper
    return decorator


# 移除 @retry_with_backoff 装饰器和 genai.Client 类型标注
def generate_llm_analysis_report(client, market_payload: dict, financial_profile: dict, session_mode: str) -> str:
    ...

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

    # 切换为获取 DEEPSEEK_API_KEY
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL", sender_email)

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

def generate_llm_analysis_report(client: OpenAI, market_payload: dict, financial_profile: dict, session_mode: str) -> str:
    """调用 DeepSeek-V3 生成策略简报，内置脱敏与重试机制"""
    
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
你是一名服务于进取型中长线投资者的买方对冲基金首席宏观与量化税务配置专家。
请严格恪守以下投资与税务纪律：
1. **默认持有立场（Default Hold）**：投资者追求较低换手率（重仓标的持仓 >1 年），绝不追高，避免杀跌。日常行情波动建议保持【按兵不动】。
2. **Tax Lots 批次级避税指令**：
   - 当特定批次持有接近 365 天（300~364天）且处于浮盈时，坚决禁止止盈建议，警示短期资本利得税惩罚；
   - 若某特定批次浮亏严重且该投资者 YTD 已实现盈利较大，评估 Tax-Loss Harvesting 冲销价值（注意提示 30 天 Wash Sale 规则）。
   - 结合报税身份、收入门槛（如 3.8% NIIT 附加税）及所在州税摩擦（Tax Friction / Tax Drag）在后台进行综合测算。
3. **关键击球点（Fat Pitch）资金管理**：
   - 基于传入的【可用现金储备】，不轻易建议开仓。在核心标的发生非理性深度回调（进入 52 周低位或 200-DMA 关键支撑）或出现不可错失的催化剂时，建议动用 5%~10% 现金分批建仓。
4. **持仓赛道暴露度与新股互补诊断**：
   - 评估当前持仓在主要赛道（如算力/半导体、大科技平台、中概互联、商业航天/核能、生物制药）的集中风险。
   - 从传入的【complementary_candidates_market_data】中挑选 1~2 只最能与当前持仓形成互补/对冲的标的（附带推荐逻辑与建议挂单/建仓区间）。
   - **严禁凭空编造价格**：建仓区间必须严格基于该标的的【current_price（当前真实价格）】、【ma_50】或【ma_200】进行计算（例如：若当前市价为 $259，建议建仓区间可为回踩 50-DMA 或折价 5%~10% 的真实计算区间，如 $235 - $245）。
5. **【关键隐私脱敏纪律】**：
   - **严禁在最终生成的 HTML 邮件正文中明文显示或打印【预估家庭年收入】与【YTD 已实现资本盈亏】的具体金额数字**！
   - 这些数值仅用于你在后台测算税率档位与止损冲销逻辑，邮件正文展示个人概况时只需显示报税身份（如 Married Filing Jointly）、报税州（NJ）及投资目标即可。
6. **输出格式**：直接输出现代、极简、多终端（PC、移动端APP）易读、带内联 CSS 样式的原生 HTML 正文，禁止包含 Markdown 代码块标记（如 ```html）。
"""

    prompt = f"""
请分析以下实时市场 Payload 与个人财务参数，生成一份【{session_title}】HTML 邮件。

【用户财务与税务配置（后台计算用参数）】
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
4. **持仓赛道集中度与新股建仓候选 (New Ticker Ideas)**：
   - 诊断当前组合的赛道集中度风险（如 AI 算力与科技股集中度）。
   - 推荐 1~2 只能够与现有持仓互补的优质新标的，列出：标的代码(Ticker)、所属赛道、互补价值与长线催化剂、建议等待的回调介入区间。
5. **基本面与新闻风险提炼**：简要概括影响长期逻辑的新闻要点，过滤短期噪音。
6. **脱敏确认**：再次确认正文中**不包含**预估年收入与 YTD 已实现盈亏的具体数字。
"""

    max_retries = 3
    delay = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Info] 正在调用 DeepSeek-V3 (deepseek-chat) 生成简报 (第 {attempt} 次)...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                stream=False
            )
            raw_text = response.choices[0].message.content
            cleaned_html = raw_text.replace("```html", "").replace("```", "").strip()
            return cleaned_html

        except Exception as e:
            print(f"[Warning] DeepSeek 调用失败 ({type(e).__name__}): {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError("DeepSeek API 重试耗尽，生成简报失败。")


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
