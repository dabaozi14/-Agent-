import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 载入环境变量 (如 API Key 和 Base URL)
load_dotenv()

# 初始化默认使用的 LLM
# 默认使用 OpenAI API。如果在 .env 中配置了 OPENAI_API_BASE 和 OPENAI_MODEL_NAME，则自动适配国内模型。
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")
llm = ChatOpenAI(
    model=model_name,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    max_tokens=2048,
    temperature=0.7
)

# ==========================================
# 步骤 1: 定义协同运营的 Agents (智能体)
# ==========================================

data_analyst = Agent(
    role='资深数据与趋势分析师',
    goal='分析最新的市场趋势、用户行为数据，并提取出核心的运营洞察。',
    backstory='你是一名在互联网和新媒体行业有10年经验的数据分析专家。你擅长从繁杂的信息和热搜库中发现极具传播潜力的爆款话题和用户痛点。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

content_creator = Agent(
    role='金牌内容运营官',
    goal='基于分析师提供的核心洞察，创作引人入胜、有网感、高转化率的社交媒体文案。',
    backstory='你是一位极具创意的爆款文案写手，深谙“小红书”、“微信公众号”等各大平台的流量密码。你的文笔幽默、接地气，且非常善于使用恰当的 Emoji 和排版来吸引读者眼球。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

operations_manager = Agent(
    role='首席运营总监 (COO)',
    goal='审核内容创作者的文案，确保其符合品牌调性，不包含风险词汇，并制定最终的发布策略。',
    backstory='你是这家公司的COO，拥有极高的审美标准和商业战略眼光。你负责把控所有的对外内容输出，确保每一次运营活动都能带来实际的商业转化，同时避免任何公关危机。',
    verbose=True,
    allow_delegation=True, # 允许把任务打回给内容运营修改
    llm=llm
)

# ==========================================
# 步骤 2: 定义协同运营的 Tasks (任务)
# ==========================================

# 示例运营主题：推广一款新的“AI办公自动化助手”产品
topic = "AI 办公自动化助手（主打功能：一键生成周报、自动回复邮件、会议纪要提取）"

task_analyze = Task(
    description=f'针对主题：【{topic}】，调查目前职场白领的最新痛点，并总结出3个最能引发共鸣的话题方向。输出必须简明扼要，直接给出洞察结论。',
    expected_output='一份包含3个核心用户痛点和话题方向的数据洞察简报。',
    agent=data_analyst
)

task_create_content = Task(
    description='根据数据分析师提供的洞察简报，撰写一篇“小红书”风格的种草文案。要求：\n1. 包含吸引眼球的吸睛标题。\n2. 正文分段清晰，合理使用 Emoji。\n3. 自然地植入产品核心卖点。\n4. 结尾加上互动引导和 3-5 个热门话题标签 (Hashtags)。',
    expected_output='一篇可以直接发布到小红书的完整图文配文（约 300 - 500 字）。',
    agent=content_creator
)

task_review_and_publish = Task(
    description='审核内容创作者提交的文案。检查其是否具备传播性、是否有语法错误或敏感词汇。如果没有问题，输出最终定稿版本，并附带一小段（约50字）的最佳发布时间及渠道投放建议。如果认为文案不够好，说明修改意见。',
    expected_output='审核通过的最终文案定稿，加上简短的发布策略建议。',
    agent=operations_manager
)

# ==========================================
# 步骤 3: 组装 Crew (团队) 并执行
# ==========================================

# 将所有 Agent 和 Task 组装为一个流水线作业的 Crew
ops_crew = Crew(
    agents=[data_analyst, content_creator, operations_manager],
    tasks=[task_analyze, task_create_content, task_review_and_publish],
    process=Process.sequential, # 顺序执行：分析 -> 创作 -> 审核
    verbose=True
)

if __name__ == "__main__":
    print(f"\n==============================================")
    print(f"🚀 开始执行多Agent协同运营任务，主题：{topic}")
    print(f"==============================================\n")
    
    # 启动任务流水线
    result = ops_crew.kickoff()
    
    print(f"\n==============================================")
    print(f"✅ 最终运营产出结果 (Final Output):")
    print(f"==============================================\n")
    print(result)
