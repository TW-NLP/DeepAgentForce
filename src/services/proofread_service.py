"""
校对服务 - 文本校对（带分块加速）
支持两种校对模式：
1. 通用 JSON 格式（默认）
2. ChineseErrorCorrector 专用格式（reasoning_content + <think>... 标签）
"""
import json
import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# 尝试导入中文纠错适配器
try:
    from src.services.chinese_corrector_adapter import (
        ChineseCorrectorAdapter,
        ChineseCorrectorSyncAdapter,
        CorrectionResult
    )
    CHINESE_CORRECTOR_AVAILABLE = True
except ImportError:
    CHINESE_CORRECTOR_AVAILABLE = False
    logger.warning("中文纠错适配器不可用，请安装 aiohttp: pip install aiohttp")

# 分块配置
CHUNK_MIN_LEN = 16
CHUNK_MAX_LEN = 1000  # 用户指定

# LLM 批量校对配置
BATCH_SIZE = 5         # 每个 LLM 调用处理的块数（越多越快但越贵）
MAX_CONCURRENCY = 15  # 最大并发数

# 子分句标点
SUB_SPLIT_FLAG = [',', '，', ';', '；', ')', '）']


def replace_blanks(text: str) -> str:
    """替换空白字符"""
    return re.sub(r'[\xa0\u3000]+', ' ', text)


def split_subsentence(sentence: str, min_len: int = CHUNK_MIN_LEN) -> List[str]:
    """按子分句标点分割长句"""
    result = []
    sent = ''
    for i, c in enumerate(sentence):
        sent += c
        if c in SUB_SPLIT_FLAG:
            # 最后两个字皆为sub_split_flag时不进行分句
            if i == len(sentence) - 2:
                result.append(sent[:-1] + c + sentence[-1])
                break

            flag = True
            for j in range(i + 1, min(len(sentence) - 1, i + 6)):
                if sentence[j] == '，' or j == len(sentence) - 1:
                    flag = False
            if (flag and len(sent) >= min_len) or i == len(sentence) - 1:
                result.append(sent[:-1] + c)
                sent = ''
        elif i == len(sentence) - 1:
            result.append(sent)
    return result


async def split_sentence(document_input: str) -> List[Tuple[int, str]]:
    """
    按标点分割句子，返回 offset 和内容列表
    """
    sent_list = []

    try:
        # 连续使用标点符号
        punctuation_flag = re.search(
            r'[^\w《》“"【】\[\]<>（）()〔〕「」『』〖〗〈〉﹛﹜{}×—－\-%％￥$□℃\xa0\u3000\r\n 　]{2,}',
            document_input
        )

        if punctuation_flag:
            document = re.sub(
                r'(?P<quotation_mark>([^\w《了么为"【】\[\]<>（）()〔〕「」『』〖〗〈〉﹛﹜{}×—－\-%％￥$□℃\xa0\u3000\r\n 　]{2,}))',
                r'\g<quotation_mark>\n', document_input)
        else:
            # 单字符断句符
            document = re.sub(r'(?P<quotation_mark>([。？！…?!|](?![了\]\'"])))', r'\g<quotation_mark>\n', document_input)
            # 特殊引号
            document = re.sub(r'(?P<quotation_mark>(([。？！!?|]|…{1,2})[了"\']))', r'\g<quotation_mark>\n', document)

        sent_list_ori = document.split('\n')
        for sent in sent_list_ori:
            sent = sent.replace('|', '')
            if not sent:
                continue

            if len(sent) > CHUNK_MAX_LEN:
                sent_list.extend(split_subsentence(sent, min_len=CHUNK_MIN_LEN))
            else:
                sent_list.append(sent)
    except Exception:
        sent_list.clear()
        sent_list.append(document_input)

    # 计算 offset
    p = 0
    res = []
    for sent in sent_list:
        res.append([p, sent])
        p += len(sent)

    return res


async def split_paragraph_lst(paragraph_lst: List[str]) -> tuple:
    """
    分割段落列表，返回 (res, res_hit)

    res: [[offset, text], ...]
    res_hit: [[offset, text], ...] 包含换行符
    """
    # 换行符预处理
    preprocessed = []
    for s in paragraph_lst:
        s = replace_blanks(s)
        s = s.replace('\r', '').split('\n')
        for s_ in s:
            s_ = s_.split('|')
            preprocessed.extend(s_)
    paragraph_lst = preprocessed

    p = 0
    offset_lst = []
    for i, s in enumerate(paragraph_lst):
        offset_lst.append(p)
        p += len(s)

    res = []
    res_hit = []
    for offset_sent, sent in zip(offset_lst, paragraph_lst):
        sent = sent.replace('|', '')
        if not sent.strip():
            continue

        if len(sent) > CHUNK_MAX_LEN:
            for offset_subsent, subsent in await split_sentence(sent):
                if not subsent.strip():
                    continue
                offset = offset_sent + offset_subsent
                res.append([offset, subsent])
                res_hit.append([offset, subsent])
        else:
            res.append([offset_sent, sent])
            res_hit.append([offset_sent, sent])
        res_hit.append([-1, '\n'])

    return res, res_hit


class ProofreadService:
    """校对服务"""

    # 常见错别字对照表
    COMMON_TYPOS = {
        '象': '像',  # 只保留 "象" -> "像" 的转换
        '目地': '目的',  # 单独添加 "目地" 这个常见错误
        '已后': '以后', '那里': '哪里', '那么': '怎么',
        '在哪': '在哪儿', '在哪': '在哪儿',
    }

    # 常见标点错误（已禁用，避免误报）
    # 如果需要启用，可以取消下面的注释
    PUNCTUATION_ERRORS = [
        # (r'[，,]\s*[，,]', '连续逗号'),
        # (r'[。。]+', '连续句号'),
        # (r'\s+[，。；：]', '标点前多余空格'),
        # (r'[，。；：]\s*$', '句尾多余标点'),
    ]

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._last_reasoning = ""
        self._corrector_adapter = None
        self._use_chinese_corrector = False

    def _init_corrector_adapter(self):
        """初始化中文纠错适配器（懒加载）"""
        if not CHINESE_CORRECTOR_AVAILABLE:
            return False

        if self._corrector_adapter is not None:
            return True

        # 获取校对专用配置
        proofread_use_dedicated = getattr(self.settings, 'PROOFREAD_USE_DEDICATED', False)
        if isinstance(proofread_use_dedicated, str):
            proofread_use_dedicated = proofread_use_dedicated.lower() in ('true', '1', 'yes')

        proofread_api_url = getattr(self.settings, 'PROOFREAD_API_URL', '') or self.settings.LLM_BASE_URL
        proofread_api_key = getattr(self.settings, 'PROOFREAD_API_KEY', '') or self.settings.LLM_API_KEY
        proofread_model = getattr(self.settings, 'PROOFREAD_MODEL', '') or self.settings.LLM_MODEL

        if not proofread_use_dedicated or not proofread_model:
            return False

        logger.info(f"[ProofreadService] 初始化 ChineseCorrector 适配器: {proofread_model}, URL: {proofread_api_url}")

        try:
            self._corrector_adapter = ChineseCorrectorSyncAdapter(
                api_url=proofread_api_url,
                api_key=proofread_api_key,
                model_name=proofread_model,
                timeout=120
            )
            self._use_chinese_corrector = True
            return True
        except Exception as e:
            logger.error(f"[ProofreadService] 初始化 ChineseCorrector 适配器失败: {e}")
            return False

    @property
    def model(self):
        if self._model is None:
            # 如果启用了校对专用模型配置，使用专用配置
            proofread_use_dedicated = getattr(self.settings, 'PROOFREAD_USE_DEDICATED', False)
            # 处理字符串形式的布尔值（从前端 checkbox 传来）
            if isinstance(proofread_use_dedicated, str):
                proofread_use_dedicated = proofread_use_dedicated.lower() in ('true', '1', 'yes')
            proofread_model = getattr(self.settings, 'PROOFREAD_MODEL', '') or self.settings.LLM_MODEL
            proofread_api_key = getattr(self.settings, 'PROOFREAD_API_KEY', '') or self.settings.LLM_API_KEY
            proofread_base_url = getattr(self.settings, 'PROOFREAD_BASE_URL', '') or self.settings.LLM_BASE_URL

            logger.info(f"[ProofreadService] 加载配置: PROOFREAD_USE_DEDICATED={proofread_use_dedicated}, PROOFREAD_MODEL={proofread_model}, PROOFREAD_BASE_URL={proofread_base_url}")

            if proofread_use_dedicated and proofread_model:
                model_name = proofread_model
                api_key = proofread_api_key
                base_url = proofread_base_url
                logger.info(f"[ProofreadService] 使用校对专用模型: {model_name}, base_url: {base_url}")
            else:
                # 否则复用通用 LLM 配置
                model_name = self.settings.LLM_MODEL
                api_key = self.settings.LLM_API_KEY
                base_url = self.settings.LLM_BASE_URL
                logger.info(f"[ProofreadService] 使用通用 LLM 配置: {model_name}, base_url: {base_url}")

            self._model = init_chat_model(
                model=model_name,
                model_provider="openai",
                api_key=api_key,
                base_url=base_url,
            )
        return self._model

    async def proofread_text(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """
        校对文本（分块并行加速）

        Args:
            text: 待校对的文本
            progress_callback: 进度回调函数，签名为 (current: int, total: int) -> None

        Returns:
            {
                "issues": [...],
                "summary": {...}
            }
        """
        issues = []

        # 1. 规则-based 基础检测（已禁用，避免重复）
        # rule_issues = self._rule_based_check(text)
        # issues.extend(rule_issues)

        # 2. 文本分块
        paragraphs = text.split('\n')
        chunks, _ = await split_paragraph_lst(paragraphs)

        # 3. 并行 LLM 校对
        try:
            # 检查是否使用中文纠错适配器
            if self._init_corrector_adapter() and self._use_chinese_corrector:
                logger.info("[ProofreadService] 使用 ChineseCorrector 适配器进行校对")
                llm_issues = await self._llm_proofread_with_chinese_corrector(chunks, text, progress_callback)
            else:
                logger.info("[ProofreadService] 使用标准 LLM 校对")
                llm_issues = await self._llm_proofread_parallel(chunks, text, progress_callback)

            # 保留 reasoning_content
            all_reasonings = [issue.get("reasoning", "") for issue in llm_issues if issue.get("reasoning")]
            self._last_reasoning = "\n".join(all_reasonings) if all_reasonings else ""
            issues.extend(llm_issues)
        except Exception as e:
            logger.error(f"LLM 校对失败: {e}")

        # 去重
        issues = self._dedupe_issues(issues)

        # 统计
        summary = self._generate_summary(issues)

        return {
            "issues": issues,
            "summary": summary
        }

    def _rule_based_check(self, text: str) -> List[Dict[str, Any]]:
        """基于规则的检查（使用全局位置）"""
        issues = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            # 计算该行在全文中的起始位置
            global_line_start = 0
            for i in range(line_num - 1):
                global_line_start += len(lines[i]) + 1

            # 检查常见错别字
            for wrong, correct in self.COMMON_TYPOS.items():
                if wrong in line:
                    pattern = re.escape(wrong)
                    for match in re.finditer(pattern, line):
                        local_start = match.start()
                        local_end = match.end()
                        issues.append({
                            "type": "error",
                            "category": "错别字",
                            "position": {
                                "line": line_num,
                                "start": global_line_start + local_start,
                                "end": global_line_start + local_end
                            },
                            "original": wrong,
                            "suggestion": correct,
                            "explanation": f"'{wrong}' 应改为 '{correct}'"
                        })

            # 检查标点错误
            for pattern, desc in self.PUNCTUATION_ERRORS:
                matches = list(re.finditer(pattern, line))
                if matches:
                    for match in matches:
                        issues.append({
                            "type": "warning",
                            "category": "标点",
                            "position": {
                                "line": line_num,
                                "start": global_line_start + match.start(),
                                "end": global_line_start + match.end()
                            },
                            "original": match.group(),
                            "suggestion": None,
                            "explanation": desc
                        })

        return issues

    async def _llm_proofread_single(self, chunk_text: str, chunk_offset: int) -> List[Dict[str, Any]]:
        """校对单个文本块"""
        prompt = f"""你是一个专业的中文校对员。请仔细检查以下文本中的问题：

文本：
---
{chunk_text}
---

请检查以下类型的问题：
1. 错别字
2. 标点符号错误
3. 用词不当
4. 语法错误
5. 格式问题

请以 JSON 格式返回结果：
{{
    "issues": [
        {{
            "type": "error",
            "category": "错别字/标点/用词/语法/格式",
            "start": 该文本块内起始位置(从0开始),
            "end": 该文本块内结束位置,
            "original": "原文",
            "suggestion": "修改建议",
            "explanation": "问题说明"
        }}
    ]
}}

重要要求：
1. start 和 end 是相对于该文本块开头的字符位置（从0开始计数）
2. original 必须是文本中完全匹配的字符串
3. 只返回 JSON，不要有其他文字。/no_think"""

        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        content = response.content

        # 提取 JSON
        try:
            import json
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            issues = data.get("issues", [])
            result = []
            for issue in issues:
                local_start = issue.get("start", 0)
                local_end = issue.get("end", 0)
                original = issue.get("original", "")

                # 转换为全局位置
                global_start = chunk_offset + local_start
                global_end = chunk_offset + local_end

                result.append({
                    "type": issue.get("type", "warning"),
                    "category": issue.get("category", ""),
                    "position": {
                        "line": 0,  # 不计算行号
                        "start": global_start,
                        "end": global_end
                    },
                    "original": original,
                    "suggestion": issue.get("suggestion"),
                    "explanation": issue.get("explanation", "")
                })
            return result
        except Exception as e:
            logger.error(f"解析 LLM 响应失败: {e}")
            return []

    async def _llm_proofread_batch(self, batch: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """一次校对多个文本块，减少 API 调用次数"""
        if not batch:
            return []

        prompt_parts = []
        for i, (offset, text) in enumerate(batch):
            prompt_parts.append(f"【块 {i + 1}】(offset={offset})\n{text}")

        prompt = f"""请分析以下文本中的错误：

{chr(10).join(prompt_parts)}

请严格按照以下格式输出（每句话一个结果）：
[纠错结果]
原句: <原文>
错情: <错误类型>，<修改原因>
错误位置：起始位置=<数字>，终止位置=<数字>
正确建议：<纠正后的文本>

如果没有错误，请输出：
[纠错结果]
原句: <原文>
错情: 无误
正确建议: <原文>

只返回纠错结果，不要有其他文字。"""

        # 使用 system prompt + user message 格式
        system_prompt = "假如你是一名专业的纠错专家，请分析输入句子的语法错误类型和修改原因，并只输出纠正后的语句，错误类型如下：错别字、词语搭配错误、词性错误、语序错误、成分残缺、成分赘余、关联词使用错误、指代不明、语义逻辑不通、无误。"

        response = await self.model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        content = response.content

        # 获取 reasoning_content（如果模型支持）
        reasoning_content = getattr(response, 'reasoning_content', None) or ""

        logger.info(f"[ProofreadService] 收到回复 content={content[:200]}...")
        logger.info(f"[ProofreadService] reasoning_content={reasoning_content[:200] if reasoning_content else 'None'}...")

        # 直接解析自定义格式（包含 <think> 和 的格式）
        return self._parse_corrector_response(content, batch, reasoning_content)

    def _parse_custom_proofread_format(self, content: str, batch: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """
        解析 ChineseErrorCorrector 模型返回的自定义格式

        格式示例：
        [纠错结果]
        原句: 我的明字叫小明
        错情: 错误类型：错别字\n修改原因：...
        正确建议：我的名字叫小明
        """
        result = []
        try:
            # 按 [纠错结果] 分割多个纠错结果
            sections = re.split(r'\[纠错结果\]', content)
            for section in sections:
                if not section.strip():
                    continue

                # 提取原句
                original_match = re.search(r'原句[:：]\s*(.+)', section)
                if not original_match:
                    continue
                original_sentence = original_match.group(1).strip()

                # 如果是"无误"，跳过
                if '无误' in section:
                    continue

                # 提取错误类型和修改原因
                category = "错别字"
                explanation = ""
                cuoqing_match = re.search(r'错情[:：]\s*(.+)', section, re.DOTALL)
                if cuoqing_match:
                    cuoqing_text = cuoqing_match.group(1).strip()
                    # 清理转义的换行符
                    cuoqing_text = cuoqing_text.replace('\\n', '\n').replace('\\N', '\n')
                    explanation = cuoqing_text

                    # 提取错误类型
                    if "错误类型" in cuoqing_text:
                        type_match = re.search(r'错误类型[:：]\s*([^，,。.\n]+)', cuoqing_text)
                        if type_match:
                            category_text = type_match.group(1).strip()
                            if "错别字" in category_text:
                                category = "错别字"
                            elif "标点" in category_text:
                                category = "标点"
                            elif "用词" in category_text or "词不当" in category_text:
                                category = "用词"
                            elif "语法" in category_text:
                                category = "语法"
                            elif "语序" in category_text:
                                category = "语序"
                            elif "成分残缺" in category_text:
                                category = "成分残缺"
                            elif "成分赘余" in category_text:
                                category = "成分赘余"
                            elif "关联词" in category_text:
                                category = "关联词"
                            elif "指代" in category_text:
                                category = "指代不明"
                            elif "语义" in category_text or "逻辑" in category_text:
                                category = "语义逻辑"

                # 提取错误位置
                start_pos, end_pos = None, None
                pos_match = re.search(r'起始位置\s*[=：:]\s*(\d+)[，,]\s*终止位置\s*[=：:]\s*(\d+)', section)
                if pos_match:
                    start_pos = int(pos_match.group(1))
                    end_pos = int(pos_match.group(2))

                # 提取正确建议
                suggestion = None
                suggestion_match = re.search(r'正确建议[:：]\s*(.+)', section, re.DOTALL)
                if suggestion_match:
                    suggestion_text = suggestion_match.group(1).strip()
                    # 排除空值和特殊标记
                    if suggestion_text and suggestion_text not in ['无', 'NULL', 'null', 'None', '']:
                        suggestion = suggestion_text

                # 如果没有找到位置，尝试在原文中查找匹配
                if start_pos is None or end_pos is None:
                    # 尝试通过正确建议来定位
                    if suggestion and suggestion != original_sentence:
                        # 找到不同之处
                        for i, (a, b) in enumerate(zip(original_sentence, suggestion)):
                            if a != b:
                                start_pos = i
                                break
                        if start_pos is not None:
                            # 计算结束位置（取原句和建议中较短的长度）
                            min_len = min(len(original_sentence), len(suggestion))
                            end_pos = min_len
                            for i in range(min_len - 1, start_pos - 1, -1):
                                if original_sentence[i] != suggestion[i]:
                                    end_pos = i + 1
                                    break

                # 找到该原句属于哪个块
                chunk_idx = None
                chunk_offset = 0
                for idx, (offset, chunk_text) in enumerate(batch):
                    if original_sentence in chunk_text:
                        chunk_idx = idx
                        chunk_offset = offset
                        break

                if chunk_idx is None:
                    # 如果找不到匹配的块，尝试模糊匹配
                    for idx, (offset, chunk_text) in enumerate(batch):
                        # 检查原句是否是块的子串
                        if original_sentence in chunk_text:
                            chunk_idx = idx
                            chunk_offset = offset
                            break

                if chunk_idx is None:
                    continue

                # 转换到全局位置
                global_start = chunk_offset + (start_pos if start_pos is not None else 0)
                global_end = chunk_offset + (end_pos if end_pos is not None else len(original_sentence))

                # 构建原始文本（需要在完整块中查找）
                chunk_text = batch[chunk_idx][1]
                original_in_chunk = original_sentence
                if original_sentence not in chunk_text:
                    # 模糊查找
                    for i in range(len(chunk_text) - len(original_sentence) + 1):
                        if chunk_text[i:i+len(original_sentence)] == original_sentence:
                            global_start = chunk_offset + i
                            global_end = global_start + len(original_sentence)
                            original_in_chunk = original_sentence
                            break

                result.append({
                    "type": "error",
                    "category": category,
                    "position": {
                        "line": 0,
                        "start": global_start,
                        "end": global_end
                    },
                    "original": original_in_chunk,
                    "suggestion": suggestion,
                    "explanation": explanation
                })

        except Exception as e:
            logger.error(f"解析自定义校对格式失败: {e}")

        return result

    async def _llm_proofread_parallel(self, chunks: List[Tuple[int, str]], text: str, progress_callback=None) -> List[Dict[str, Any]]:
        """并行校对所有文本块（批量模式减少 API 调用）"""
        if not chunks:
            return []

        batches = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batches.append(chunks[i:i + BATCH_SIZE])

        total = len(batches)
        completed = 0
        logger.info(f"文本校对：共 {len(chunks)} 个块，分 {total} 批，每批最多 {BATCH_SIZE} 个")

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def proofread_batch(batch):
            nonlocal completed
            async with semaphore:
                result = await self._llm_proofread_batch(batch)
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total)
                    except Exception:
                        pass
                return result

        tasks = [proofread_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_issues = []
        for result in results:
            if isinstance(result, list):
                all_issues.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"批量校对异常: {result}")

        return all_issues

    def _parse_corrector_response(self, content: str, batch: List[Tuple[int, str]], reasoning_content: str = "") -> List[Dict[str, Any]]:
        """
        解析 ChineseErrorCorrector 模型返回的内容

        格式示例：
        <think>
        错误类型：错别字
        修改原因：...

        我的名字叫小明

        或者：
        [纠错结果]
        原句: 我的明字叫小明
        错情: 错误类型：错别字，修改原因：...
        正确建议：我的名字叫小明
        """
        result = []
        reasoning = reasoning_content

        # 清理 <think> 和  标记
        clean_content = content
        if "<think>" in content and "" in content:
            think_match = re.search(r'<think>\s*(.*?)\s*\s*(.+)', content, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                clean_content = think_match.group(2).strip()

        # 处理多个句子的结果（用换行分隔或多个 <think>... 块）
        sentences = []

        # 如果 clean_content 是纠正后的单个句子（没有格式），则按句子分割处理
        lines = clean_content.split('\n')
        current_sentence = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 如果包含格式标记，则是新的纠错结果
            if '原句' in line or '正确建议' in line or '错情' in line or '[纠错结果]' in line:
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = line
            else:
                current_sentence += "\n" + line if current_sentence else line
        if current_sentence:
            sentences.append(current_sentence)

        # 如果没有解析到格式，尝试按每行作为一个句子
        if not sentences or len(sentences) == 0:
            for line in clean_content.split('\n'):
                line = line.strip()
                if line:
                    sentences.append(line)

        logger.info(f"[ProofreadService] 解析到 {len(sentences)} 个句子")

        # 合并所有原始文本
        full_text = "\n".join([chunk[1] for chunk in batch])

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            category = "错别字"
            explanation = reasoning
            original_sentence = ""
            suggestion = None
            start_pos, end_pos = None, None

            # 格式1: 原句: xxx\n错情: xxx\n正确建议: xxx
            if '原句' in sentence:
                original_match = re.search(r'原句[:：]\s*(.+)', sentence)
                if original_match:
                    original_sentence = original_match.group(1).strip()

                cuoqing_match = re.search(r'错情[:：]\s*(.+)', sentence, re.DOTALL)
                if cuoqing_match:
                    cuoqing_text = cuoqing_match.group(1).strip().replace('\\n', '\n')
                    explanation = cuoqing_text
                    if "错误类型" in cuoqing_text:
                        type_match = re.search(r'错误类型[:：]\s*([^，,。.\n]+)', cuoqing_text)
                        if type_match:
                            category_text = type_match.group(1).strip()
                            category = self._map_category(category_text)

                suggestion_match = re.search(r'正确建议[:：]\s*(.+)', sentence, re.DOTALL)
                if suggestion_match:
                    suggestion_text = suggestion_match.group(1).strip()
                    if suggestion_text and suggestion_text not in ['无', 'NULL', 'null', 'None', '']:
                        suggestion = suggestion_text

                pos_match = re.search(r'起始位置\s*[=：:]\s*(\d+)[，,]\s*终止位置\s*[=：:]\s*(\d+)', sentence)
                if pos_match:
                    start_pos = int(pos_match.group(1))
                    end_pos = int(pos_match.group(2))
            else:
                # 格式2: 纯纠正后的句子
                suggestion = sentence
                if reasoning:
                    if "错别字" in reasoning:
                        category = "错别字"
                    elif "标点" in reasoning:
                        category = "标点"
                    elif "语法" in reasoning:
                        category = "语法"
                    elif "用词" in reasoning or "词不当" in reasoning:
                        category = "用词"
                    if "修改原因" in reasoning:
                        reason_match = re.search(r'修改原因[:：]\s*(.+)', reasoning, re.DOTALL)
                        if reason_match:
                            explanation = reason_match.group(1).strip().replace('\\n', '\n')

            if not original_sentence:
                for offset, chunk_text in batch:
                    for line in chunk_text.split('\n'):
                        line = line.strip()
                        if line and len(line) > 2:
                            if suggestion and line in full_text:
                                original_sentence = line
                                break
                    if original_sentence:
                        break

            if original_sentence and original_sentence in full_text:
                global_start = full_text.find(original_sentence)
                global_end = global_start + len(original_sentence)
            else:
                global_start = 0
                global_end = len(full_text) if i == 0 else 0

            result.append({
                "type": "error",
                "category": category,
                "position": {
                    "line": 0,
                    "start": global_start,
                    "end": global_end
                },
                "original": original_sentence or (full_text[global_start:global_end] if global_end > global_start else ""),
                "suggestion": suggestion,
                "explanation": explanation,
                "reasoning": reasoning if reasoning else ""
            })

        return result

    async def _llm_proofread_with_chinese_corrector(
        self,
        chunks: List[Tuple[int, str]],
        full_text: str,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        使用 ChineseCorrector 适配器进行校对
        专门处理包含 reasoning_content 和 <think>...  标签的响应格式
        """
        if not chunks:
            return []

        if not CHINESE_CORRECTOR_AVAILABLE or self._corrector_adapter is None:
            logger.warning("[ProofreadService] ChineseCorrector 不可用，回退到标准 LLM 校对")
            return await self._llm_proofread_parallel(chunks, full_text, progress_callback)

        all_issues = []
        total = len(chunks)
        completed = 0

        logger.info(f"[ProofreadService] ChineseCorrector 校对：共 {total} 个块")

        # 按批次处理
        for i, (offset, chunk_text) in enumerate(chunks):
            try:
                # 提取句子（按换行分割）
                sentences = [s.strip() for s in chunk_text.split('\n') if s.strip()]

                if not sentences:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    continue

                # 计算 sentences 中每个句子在 chunk_text 中的位置
                sentence_positions = []
                temp_pos = 0
                for s in chunk_text.split('\n'):
                    stripped = s.strip()
                    if stripped:
                        # 找到该句子在 chunk_text 中的实际位置
                        pos = chunk_text.find(stripped, temp_pos)
                        if pos >= 0:
                            sentence_positions.append(pos)
                            temp_pos = pos + len(stripped)

                # 调用 ChineseCorrector
                results = self._corrector_adapter.correct(sentences)

                # 构建 original -> position 的映射
                orig_to_pos = {}
                for i, sent in enumerate(sentences):
                    if i < len(sentence_positions):
                        orig_to_pos[sent] = sentence_positions[i]

                # 转换结果为 issues - 只保留精确的 diff span
                for ri, result in enumerate(results):
                    if not isinstance(result, CorrectionResult):
                        continue

                    # 跳过无错误的情况
                    if result.original == result.corrected and not result.error_type:
                        continue

                    # 跳过没有 diff_spans 的情况（无法确定精确位置）
                    if not result.diff_spans:
                        continue

                    # 构建 explanation
                    explanation = result.explanation
                    if not explanation and result.error_type:
                        explanation = f"错误类型：{result.error_type}"

                    # 计算原句在 chunk_text 中的位置
                    base_pos = -1

                    # 方法1: 直接从映射中查找
                    if result.original in orig_to_pos:
                        base_pos = orig_to_pos[result.original]
                    # 方法2: 如果 ri 在范围内，检查 sentences[ri] 是否匹配
                    elif ri < len(sentences) and sentences[ri] == result.original:
                        if ri < len(sentence_positions):
                            base_pos = sentence_positions[ri]
                    # 方法3: 模糊匹配
                    else:
                        for si, sp in enumerate(sentence_positions):
                            s_text = sentences[si] if si < len(sentences) else ""
                            if s_text == result.original:
                                base_pos = sp
                                break
                        # 方法4: 检查是否包含关系
                        if base_pos < 0:
                            for si, sp in enumerate(sentence_positions):
                                s_text = sentences[si] if si < len(sentences) else ""
                                if result.original in s_text or s_text in result.original:
                                    base_pos = sp
                                    break

                    if base_pos < 0:
                        # 无法确定位置，跳过
                        logger.warning(f"[ProofreadService] 无法确定句子位置，跳过: '{result.original[:30]}...'")
                        continue

                    # 验证 diff_spans 的位置是否合理
                    valid_spans = []
                    for span in result.diff_spans:
                        # 位置必须非负
                        if span.start < 0:
                            continue
                        # end 必须大于等于 start
                        if span.end < span.start:
                            continue
                        # 对于删除/替换操作，end 不能超过 original 长度
                        if span.original_text and span.end > len(result.original):
                            continue
                        valid_spans.append(span)

                    if not valid_spans:
                        continue

                    # 为每个 diff span 生成一个 issue
                    for span in valid_spans:
                        start = offset + base_pos + span.start
                        end = offset + base_pos + span.end

                        # 对于删除操作，suggestion 为空
                        suggestion = span.corrected_text if span.corrected_text else None

                        issue = {
                            "type": "error",
                            "category": result.error_type or "错别字",
                            "position": {
                                "line": 0,
                                "start": start,
                                "end": end
                            },
                            "original": span.original_text,
                            "suggestion": suggestion,
                            "explanation": explanation,
                            "reasoning": result.reasoning
                        }
                        all_issues.append(issue)
                        logger.info(f"[ProofreadService] 生成 issue: orig='{span.original_text}', corr='{span.corrected_text}', pos={start}-{end}")

            except Exception as e:
                logger.error(f"[ProofreadService] ChineseCorrector 校对块 {i} 失败: {e}")

            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total)
                except Exception:
                    pass

        return all_issues

    async def _llm_proofread_batch_with_chinese_corrector(self, batch: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """
        使用 ChineseCorrector 适配器处理单个批次
        """
        if not batch:
            return []

        if not CHINESE_CORRECTOR_AVAILABLE or self._corrector_adapter is None:
            logger.warning("[ProofreadService] ChineseCorrector 不可用，使用标准 LLM")
            return await self._llm_proofread_batch(batch)

        all_issues = []
        full_text = "\n".join([chunk[1] for chunk in batch])

        for offset, chunk_text in batch:
            try:
                sentences = [s.strip() for s in chunk_text.split('\n') if s.strip()]
                if not sentences:
                    continue

                # 计算 sentences 中每个句子在 chunk_text 中的位置
                sentence_positions = []
                temp_pos = 0
                for s in chunk_text.split('\n'):
                    stripped = s.strip()
                    if stripped:
                        pos = chunk_text.find(stripped, temp_pos)
                        if pos >= 0:
                            sentence_positions.append(pos)
                            temp_pos = pos + len(stripped)

                results = self._corrector_adapter.correct(sentences)

                # 构建 original -> position 的映射
                orig_to_pos = {}
                for i, sent in enumerate(sentences):
                    if i < len(sentence_positions):
                        orig_to_pos[sent] = sentence_positions[i]

                for ri, result in enumerate(results):
                    if not isinstance(result, CorrectionResult):
                        continue

                    if result.original == result.corrected and not result.error_type:
                        continue

                    explanation = result.explanation
                    if not explanation and result.error_type:
                        explanation = f"错误类型：{result.error_type}"

                    # 跳过没有 diff_spans 的情况
                    if not result.diff_spans:
                        continue

                    # 计算原句在 chunk_text 中的位置
                    base_pos = -1

                    # 方法1: 直接从映射中查找
                    if result.original in orig_to_pos:
                        base_pos = orig_to_pos[result.original]
                    # 方法2: 如果 ri 在范围内，检查 sentences[ri] 是否匹配
                    elif ri < len(sentences) and sentences[ri] == result.original:
                        if ri < len(sentence_positions):
                            base_pos = sentence_positions[ri]
                    # 方法3: 模糊匹配
                    else:
                        for si, sp in enumerate(sentence_positions):
                            s_text = sentences[si] if si < len(sentences) else ""
                            if s_text == result.original:
                                base_pos = sp
                                break
                        # 方法4: 检查是否包含关系
                        if base_pos < 0:
                            for si, sp in enumerate(sentence_positions):
                                s_text = sentences[si] if si < len(sentences) else ""
                                if result.original in s_text or s_text in result.original:
                                    base_pos = sp
                                    break

                    if base_pos < 0:
                        logger.warning(f"[ProofreadService] 无法确定句子位置，跳过: '{result.original[:30]}...'")
                        continue

                    # 验证 diff_spans 的位置是否合理
                    valid_spans = []
                    for span in result.diff_spans:
                        # 位置必须非负
                        if span.start < 0:
                            continue
                        # end 必须大于等于 start
                        if span.end < span.start:
                            continue
                        # 对于删除/替换操作，end 不能超过 original 长度
                        if span.original_text and span.end > len(result.original):
                            continue
                        valid_spans.append(span)

                    if not valid_spans:
                        continue

                    # 为每个 diff span 生成一个 issue
                    for span in valid_spans:
                        start = offset + base_pos + span.start
                        end = offset + base_pos + span.end
                        suggestion = span.corrected_text if span.corrected_text else None

                        issue = {
                            "type": "error",
                            "category": result.error_type or "错别字",
                            "position": {"line": 0, "start": start, "end": end},
                            "original": span.original_text,
                            "suggestion": suggestion,
                            "explanation": explanation,
                            "reasoning": result.reasoning
                        }
                        all_issues.append(issue)
                        logger.info(f"[ProofreadService] 批次生成 issue: orig='{span.original_text}', corr='{span.corrected_text}', pos={start}-{end}")

            except Exception as e:
                logger.error(f"[ProofreadService] ChineseCorrector 批次处理失败: {e}")

        return all_issues

    def _find_in_full_text(
        self,
        full_text: str,
        original: str,
        corrected: str
    ) -> Tuple[int, int]:
        """在完整文本中查找句子的位置"""
        # 优先匹配原始句子
        if original and original in full_text:
            start = full_text.find(original)
            return (start, start + len(original))

        # 尝试匹配纠正后的句子
        if corrected and corrected in full_text:
            start = full_text.find(corrected)
            return (start, start + len(corrected))

        # 尝试模糊匹配
        if original:
            # 查找相似的句子
            for i in range(len(full_text) - len(original) + 1):
                if full_text[i:i+len(original)] == original:
                    return (i, i + len(original))

        return (0, 0)

    def _map_category(self, category_text: str) -> str:
        """映射错误类型"""
        if "错别字" in category_text:
            return "错别字"
        elif "标点" in category_text:
            return "标点"
        elif "用词" in category_text or "词不当" in category_text or "词语搭配" in category_text:
            return "用词"
        elif "语法" in category_text:
            return "语法"
        elif "语序" in category_text:
            return "语序"
        elif "成分残缺" in category_text:
            return "成分残缺"
        elif "成分赘余" in category_text:
            return "成分赘余"
        elif "关联词" in category_text:
            return "关联词"
        elif "指代" in category_text:
            return "指代不明"
        elif "语义" in category_text or "逻辑" in category_text:
            return "语义逻辑"
        return "错别字"

    async def _llm_proofread(self, text: str) -> List[Dict[str, Any]]:
        """使用 LLM 进行智能校对（兼容旧接口）"""
        return await self._llm_proofread_single(text, 0)

    def _dedupe_issues(self, issues: List[Dict]) -> List[Dict]:
        """去重并合并重叠的 issue"""
        # 按 start 升序、end 降序排序
        sorted_issues = sorted(issues, key=lambda x: (x.get("position", {}).get("start", 0), -(x.get("position", {}).get("end", 0))))

        result = []
        for issue in sorted_issues:
            pos = issue.get("position", {})
            start = pos.get("start", 0)
            end = pos.get("end", 0)
            original = issue.get("original", "")

            # 检查是否被已有的 issue 包含
            is_contained = False
            for existing in result:
                e_pos = existing.get("position", {})
                e_start = e_pos.get("start", 0)
                e_end = e_pos.get("end", 0)

                # 如果已有的 issue 包含当前 issue，跳过
                if e_start <= start and end <= e_end:
                    is_contained = True
                    break

                # 如果 start 相同但 end 更小（范围更大），替换已有的
                if start == e_start and end > e_end:
                    result.remove(existing)
                    break

                # 如果 original 相同但位置不同，保留较短的（更精确的）
                if original == existing.get("original", "") and start != e_start:
                    # 计算重叠程度
                    overlap_start = max(start, e_start)
                    overlap_end = min(end, e_end)
                    if overlap_start < overlap_end:
                        # 有重叠，保留较短的
                        if (end - start) < (e_end - e_start):
                            result.remove(existing)
                        else:
                            is_contained = True
                        break

            if not is_contained:
                result.append(issue)

        return result

    def _generate_summary(self, issues: List[Dict]) -> Dict[str, Any]:
        """生成摘要"""
        errors = sum(1 for i in issues if i.get("type") == "error")
        warnings = sum(1 for i in issues if i.get("type") == "warning")
        suggestions = sum(1 for i in issues if i.get("type") == "suggestion")

        if errors == 0 and warnings == 0 and suggestions == 0:
            overall = "文本质量良好，未检测到问题。"
        elif errors > 0:
            overall = f"检测到 {errors} 个错误需要修正，建议优先处理。"
        elif warnings > 0:
            overall = f"检测到 {warnings} 个警告，建议检查。"
        else:
            overall = f"检测到 {suggestions} 个改进建议。"

        return {
            "total_issues": len(issues),
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "overall": overall
        }

    async def proofread_document(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        校对文档（与文本校对相同）
        """
        return await self.proofread_text(text)


# 全局实例
_proofread_service: Optional[ProofreadService] = None


def get_proofread_service(settings) -> ProofreadService:
    """获取校对服务实例"""
    global _proofread_service
    if _proofread_service is None:
        _proofread_service = ProofreadService(settings)
    return _proofread_service
