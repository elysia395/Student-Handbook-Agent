from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
from tavily import TavilyClient
import json
import re

# ======================
# RAG 模块（懒加载，只有调用手册查询时才初始化）
# ======================
_retriever = None
_conversation_history = []
MAX_HISTORY_TURNS = 10

def get_retriever():
    global _retriever
    if _retriever is None:
        from RAG.retriever import Retriever
        _retriever = Retriever()
    return _retriever


def reset_history():
    global _conversation_history
    _conversation_history = []


# ======================
# 初始化配置
# ======================
load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# ======================
# 常量定义
# ======================
MAX_ITERATIONS = 10  #最大迭代次数


# ======================
# 工具1：手册查询（RAG）
# ======================
def query_handbook(question: str) -> str:
    try:
        if not question or not question.strip():
            return "错误：查询问题不能为空"

        retriever = get_retriever()
        results = retriever.search(question.strip())

        if not results:
            return "没有找到相关的手册内容"

        output = "[来源：广州中医药大学学生手册]\n\n以下是学生手册中的相关内容：\n\n"
        for i, chunk in enumerate(results):
            output += f"第 {i+1} 段：\n{chunk}\n\n"
        return output

    except Exception as e:
        return f"查询手册失败：{str(e)}"


# ======================
# 工具2：全网搜索
# ======================
def web_search(query: str) -> str:
    try:
        if not query or not query.strip():
            return "错误：搜索关键词不能为空"

        result = tavily_client.search(
            query=query.strip(),
            search_depth="basic",
            max_results=2
        )
        output = "[来源：互联网搜索]\n\n搜索结果：\n"
        for idx, item in enumerate(result["results"]):
            output += f"{idx + 1}. {item['title']}\n{item['content']}\n\n"
        return output
    except Exception as e:
        return f"搜索失败：请稍后重试"


# ======================
# 工具定义
# ======================
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_handbook",
            "description": "查询学生手册中的规章制度，比如转专业条件、奖学金评定、学分要求、请假流程、违纪处分等",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于学生手册的具体问题"
                    }
                },
                "required": ["question"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "全网搜索校园相关信息，比如新闻、通知、活动、学术资源等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"],
            }
        }
    }
]


# ======================
# 补全参数
# ======================
def ask_user_for_missing_params(tool_name, tool_args):
    if tool_name == "query_handbook" and not tool_args.get("question"):
        question = input("🤖 请问你想查询手册中的什么内容？")
        tool_args["question"] = question

    if tool_name == "web_search" and not tool_args.get("query"):
        query = input("🤖 请问你想搜索什么校园信息？")
        tool_args["query"] = query

    return tool_args


# ======================
# System Prompt
# ======================
SYSTEM_PROMPT = """你是广州中医药大学的校园智能助手。
学校基本信息：肇始于1924年，1956年经国务院批准成立广州中医学院，1995年更名为广州中医药大学。
当用户说"本校"、"我校"、"学校"时，均指广州中医药大学。

规则：
1. 如果用户的问题涉及学校规章制度（如转专业、奖学金、学分、请假、处分等），调用 query_handbook
2. 如果用户想了解校园新闻、通知、活动等实时信息，调用 web_search
3. 如果参数不完整，请明确告诉用户缺少什么信息
4. 每次只调用一个工具
5. 收到工具结果后，用简洁友好的语言回答用户
6. 不要重复询问已经提供的信息
7. 回答必须使用纯文本，绝对不要使用任何 markdown 符号，包括但不限于：**加粗**、# 标题、--- 分隔线、1. 有序列表、- 无序列表、> 引用、`代码`、[链接](url)，用自然语言分段组织回答
8. 回答末尾标注信息来源：(来源：学生手册) 或 (来源：网络搜索)"""


def strip_markdown(text: str) -> str:
    PLACEHOLDER_START = "\uE000"
    PLACEHOLDER_END = "\uE001"
    sources = re.findall(r'\[来源：[^\]]+\]', text)
    for i, s in enumerate(sources):
        text = text.replace(s, f"{PLACEHOLDER_START}SRC{i}{PLACEHOLDER_END}")
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    for i, s in enumerate(sources):
        text = text.replace(f"{PLACEHOLDER_START}SRC{i}{PLACEHOLDER_END}", s)
    return text.strip()


# ======================
# Agent 主逻辑
# ======================
def run_agent(user_input: str):
    global _conversation_history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_conversation_history)
    messages.append({"role": "user", "content": user_input})

    used_tools = set()
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1

        try:
            resp = llm_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            return f"抱歉，调用 AI 服务失败：{str(e)}"

        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                tool_args = ask_user_for_missing_params(tool_name, tool_args)

                if tool_name == "query_handbook":
                    result = query_handbook(tool_args["question"])
                    used_tools.add("handbook")
                elif tool_name == "web_search":
                    result = web_search(tool_args["query"])
                    used_tools.add("web")
                else:
                    result = "未知工具"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
        else:
            answer = strip_markdown(msg.content.strip())

            if "handbook" in used_tools:
                answer += " (来源：学生手册)"
            elif "web" in used_tools:
                answer += " (来源：网络搜索)"

            _conversation_history.append({"role": "user", "content": user_input})
            _conversation_history.append({"role": "assistant", "content": answer})

            max_messages = MAX_HISTORY_TURNS * 2
            if len(_conversation_history) > max_messages:
                _conversation_history = _conversation_history[-max_messages:]

            return answer

    return "抱歉，处理您的请求时超过了最大迭代次数，请简化您的问题。"


# ======================
# 测试（聊天循环）
# ======================
if __name__ == "__main__":
    print("🤖 校园助手已启动（输入 exit 退出，/reset 清空对话历史）\n")
    while True:
        try:
            user_input = input("你：").strip()
            if not user_input:
                continue
            if user_input.strip().lower() == "exit":
                print("🤖 再见！")
                break
            if user_input.strip().lower() == "/reset":
                reset_history()
                print("🤖 对话历史已清空\n")
                continue
            answer = run_agent(user_input)
            print("🤖 助手：", answer, "\n")
        except KeyboardInterrupt:
            print("\n🤖 再见！")
            break
        except EOFError:
            print("\n🤖 再见！")
            break
