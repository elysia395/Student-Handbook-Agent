# RAG 管线测试问题集

## 分类说明

| 类别 | 场景 | 验证目标 |
|------|------|----------|
| 基础召回 | 常规查询 | 向量粗召 + 初筛 |
| 多碎片聚合 | 需多个 chunk 回答 | 断崖截取 |
| 无关过滤 | 无关/模糊查询 | 两级兜底逻辑 |
| 边界测试 | 极限场景 | 各兜底线 |
| 系统提示词 | "本校" 指代 | System prompt 校名配置 |

---

## 一、基础召回（应返回 ≥3 条）

```
处分种类有哪些
广州中医药大学的建校时间
奖学金评选条件是什么
请假流程怎么走
考试违规如何认定
转专业需要什么条件
学分绩点怎么计算
违纪处分的申诉流程
校内勤工助学岗位申请
入党积极分子培养要求
```

## 二、多碎片聚合（需 ≥5 条才能完整回答）

```
综合测评的加分项目有哪些
学生违纪的处分等级从轻到重有哪些
学位授予的条件和要求
国家奖学金的申请条件和评审流程
学生宿舍管理规定
```

## 三、无关过滤（应返回 0 条或极少）

```
今天天气怎么样
量子力学的基本原理
怎么做红烧肉
Windows 系统怎么安装
特朗普的竞选政策
```

## 四、边界测试

```
# 向量兜底触发（相似度 0.35-0.50）
体育表现分怎么计算

# Rerank 兜底触发（极低相关性）
学生手册的封面设计理念

# 短查询（1-2 字）
处分
奖学金
放假
学费
实习

# 长查询（噪声掺杂）
广州中医药大学的学生如果考试作弊被抓到会有什么处分
请问我们学校关于奖学金评选的具体流程和加分标准是什么
```

## 五、系统提示词测试

```
本校的建校时间
我校的校训是什么
学校是什么时候成立的
我们学校的地址在哪里
本校的学生守则有哪些规定
```

## 六、快速验证脚本

```bash
# 单条测试
$env:PYTHONIOENCODING='utf-8'; D:\AI-Agent-Learning\make-agent\.venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from RAG.retriever import Retriever
r = Retriever()
q = '处分种类有哪些'
results = r.search(q)
print(f'返回 {len(results)} 条')
for i, doc in enumerate(results):
    print(f'[{i}] {doc[:120]}...')
"

# 批量测试
$env:PYTHONIOENCODING='utf-8'; D:\AI-Agent-Learning\make-agent\.venv\Scripts\python.exe -c "
import sys, time; sys.path.insert(0, '.')
from RAG.retriever import Retriever
r = Retriever()
queries = ['处分种类', '建校时间', '奖学金加分', '请假流程', '今天天气']
for q in queries:
    t0 = time.time()
    n = len(r.search(q))
    print(f'{q:15s} -> {n} 条, {time.time()-t0:.1f}s')
"
```
