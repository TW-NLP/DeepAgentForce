"""
校对服务 - 文本校对（带分块加速）
"""
import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

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
        '象': '像', '做': '作', '的地得': '的', '的得地': '得',
        '已后': '以后', '那里': '哪里', '那么': '怎么',
        '在哪': '在哪儿', '在哪': '在哪儿',
    }

    # 常见标点错误
    PUNCTUATION_ERRORS = [
        (r'[，,]\s*[，,]', '连续逗号'),
        (r'[。。]+', '连续句号'),
        (r'\s+[，。；：]', '标点前多余空格'),
        (r'[，。；：]\s*$', '句尾多余标点'),
    ]

    def __init__(self, settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = init_chat_model(
                model=self.settings.LLM_MODEL,
                model_provider="openai",
                api_key=self.settings.LLM_API_KEY,
                base_url=self.settings.LLM_BASE_URL,
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

        # 1. 规则-based 基础检测
        rule_issues = self._rule_based_check(text)
        issues.extend(rule_issues)

        # 2. 文本分块
        paragraphs = text.split('\n')
        chunks, _ = await split_paragraph_lst(paragraphs)

        # 3. 并行 LLM 校对
        try:
            llm_issues = await self._llm_proofread_parallel(chunks, text, progress_callback)
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

        prompt = f"""你是一个专业的中文校对员。请仔细检查以下文本中的问题：

---
{"=" * 50}
{chr(10).join(prompt_parts)}
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
            "chunk_idx": 块编号(从0开始),
            "start": 该块内起始位置(从0开始),
            "end": 该块内结束位置,
            "original": "原文",
            "suggestion": "修改建议",
            "explanation": "问题说明",
            "type": "error/warning/suggestion",
            "category": "错别字/标点/用词/语法/格式"
        }}
    ]
}}

重要要求：
1. chunk_idx 是块编号，对应上面【块 N】中的 N-1
2. start 和 end 是相对于该块开头的字符位置（从0开始计数）
3. original 必须是文本中完全匹配的字符串
4. 只返回 JSON，不要有其他文字。/no_think"""

        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        content = response.content

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
                chunk_idx = issue.get("chunk_idx", 0)
                if chunk_idx >= len(batch):
                    continue
                chunk_offset = batch[chunk_idx][0]
                chunk_text = batch[chunk_idx][1]

                local_start = issue.get("start", 0)
                local_end = issue.get("end", 0)

                if local_start >= len(chunk_text) or local_end > len(chunk_text):
                    continue

                global_start = chunk_offset + local_start
                global_end = chunk_offset + local_end

                result.append({
                    "type": issue.get("type", "warning"),
                    "category": issue.get("category", ""),
                    "position": {
                        "line": 0,
                        "start": global_start,
                        "end": global_end
                    },
                    "original": issue.get("original", ""),
                    "suggestion": issue.get("suggestion"),
                    "explanation": issue.get("explanation", "")
                })
            return result
        except Exception as e:
            logger.error(f"批量解析 LLM 响应失败: {e}")
            return []

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

    async def _llm_proofread(self, text: str) -> List[Dict[str, Any]]:
        """使用 LLM 进行智能校对（兼容旧接口）"""
        return await self._llm_proofread_single(text, 0)

    def _dedupe_issues(self, issues: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        result = []
        for issue in issues:
            key = (
                issue.get("position", {}).get("start"),
                issue.get("position", {}).get("end"),
                issue.get("original"),
                issue.get("category")
            )
            if key not in seen:
                seen.add(key)
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
