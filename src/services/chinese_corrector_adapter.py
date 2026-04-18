"""
中文纠错 API 适配器

专门适配 ChineseErrorCorrector 格式的 API：
- 请求格式：标准 OpenAI Chat Completions 格式
- 响应格式：content 中包含推理过程和纠正后的文本

示例响应 content：
错误类型：错别字
修改原因：原句中的"明子"应为"名字"...
【块 1】(offset=0)
我的名字叫小明
"""

import json
import re
import logging
import difflib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

def _debug(msg: str):
    logger.debug(f"[ChineseCorrector] {msg}")


def _is_meaningful_diff(original: str, corrected: str) -> bool:
    """
    判断 diff 是否是有意义的修改，而非仅仅是标点/空白差异。
    """
    if original == corrected:
        return False
    s_orig = original.strip('。！？，,.!? \t\n\r')
    s_corr = corrected.strip('。！？，,.!? \t\n\r')
    if s_orig == s_corr:
        return False
    return True


@dataclass
class DiffSpan:
    """表示原文和纠正后文本之间的差异片段"""
    start: int
    end: int
    original_text: str
    corrected_text: str


def _strip_edge_punct(text: str) -> Tuple[str, int]:
    """去掉首尾标点符号，返回 (去标点后的文本, 去掉的字符数)。"""
    if not text:
        return text, 0
    leading_punct = len(text) - len(text.lstrip(' \t\n\r'))
    trailing_punct = len(text) - len(text.rstrip(' \t\n\r'))
    pure = text[leading_punct:len(text)-trailing_punct if trailing_punct else None]
    return pure, leading_punct


def compute_diff_spans(original: str, corrected: str) -> List[DiffSpan]:
    """
    使用 difflib.SequenceMatcher.get_opcodes() 计算原文和纠正后文本之间的差异片段。
    
    tag 可以是:
    - 'equal': 两边相同
    - 'replace': 原文有内容被替换
    - 'insert': 原文缺失内容（插入）
    - 'delete': 原文多余内容（删除）
    """
    if original == corrected:
        return []

    matcher = difflib.SequenceMatcher(None, original, corrected)
    spans = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        if tag == 'replace':
            # 原文 [i1:i2] 被替换为纠正 [j1:j2]
            orig_text = original[i1:i2]
            corr_text = corrected[j1:j2]
            if _is_meaningful_diff(orig_text, corr_text):
                spans.append(DiffSpan(
                    start=i1,
                    end=i2,
                    original_text=orig_text,
                    corrected_text=corr_text
                ))

        elif tag == 'insert':
            # 原文缺失，纠正中插入 [j1:j2]
            corr_text = corrected[j1:j2]
            if _is_meaningful_diff('', corr_text):
                spans.append(DiffSpan(
                    start=i1,  # 插入位置
                    end=i1,
                    original_text='',
                    corrected_text=corr_text
                ))

        elif tag == 'delete':
            # 原文多余，被删除 [i1:i2]
            orig_text = original[i1:i2]
            if _is_meaningful_diff(orig_text, ''):
                spans.append(DiffSpan(
                    start=i1,
                    end=i2,
                    original_text=orig_text,
                    corrected_text=''
                ))

    return spans


def build_diff_spans(original: str, corrected: str) -> List[DiffSpan]:
    """
    入口函数：处理首尾标点，计算 diff span。
    """
    if original == corrected:
        return []

    s_orig, leading = _strip_edge_punct(original)
    s_corr, _ = _strip_edge_punct(corrected)

    if s_orig == s_corr:
        return []

    raw_spans = compute_diff_spans(s_orig, s_corr)

    if not raw_spans:
        return []

    return [
        DiffSpan(
            start=sp.start + leading,
            end=sp.end + leading,
            original_text=sp.original_text,
            corrected_text=sp.corrected_text,
        )
        for sp in raw_spans
    ]


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    error_type: str
    explanation: str
    reasoning: str
    diff_spans: List[DiffSpan] = field(default_factory=list)

    @property
    def positions(self) -> List[Tuple[int, int]]:
        return [(s.start, s.end) for s in self.diff_spans]

    def to_issue_dict(self, chunk_offset: int = 0) -> Dict[str, Any]:
        if self.diff_spans:
            return self._make_issue(self.diff_spans[0], chunk_offset)
        return {
            "type": "error",
            "category": self.error_type,
            "position": {"line": 0, "start": chunk_offset, "end": chunk_offset + len(self.original)},
            "original": self.original,
            "suggestion": self.corrected,
            "explanation": self.explanation,
            "reasoning": self.reasoning,
        }

    def to_issue_dicts(self, chunk_offset: int = 0) -> List[Dict[str, Any]]:
        return [self._make_issue(sp, chunk_offset) for sp in self.diff_spans]

    def _make_issue(self, span: DiffSpan, chunk_offset: int) -> Dict[str, Any]:
        return {
            "type": "error",
            "category": self.error_type,
            "position": {
                "line": 0,
                "start": chunk_offset + span.start,
                "end": chunk_offset + span.end,
            },
            "original": span.original_text,
            "suggestion": span.corrected_text,
            "explanation": self.explanation,
            "reasoning": self.reasoning,
        }


class ChineseCorrectorAdapter:
    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: int = 120):
        self.api_url = api_url.rstrip("/")
        if "/chat/completions" not in self.api_url:
            self.api_url = f"{self.api_url}/chat/completions"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    async def correct(self, texts: List[str], system_prompt: Optional[str] = None) -> List[CorrectionResult]:
        if not texts:
            return []
        system_content = system_prompt or "假如你是一名专业的纠错专家，请分析输入句子的语法错误类型和修改原因，并只输出纠正后的语句，错误类型如下：错别字、词语搭配错误、词性错误、语序错误、成分残缺、成分赘余、关联词使用错误、指代不明、语义逻辑不通、无误。"
        user_content = "\n".join([text.strip() for text in texts if text.strip()])
        messages = [{"role": "user", "content": user_content}]
        response_data = await self._call_api(messages, system_content)
        return self._parse_response(response_data, texts)

    async def correct_single(self, text: str, system_prompt: Optional[str] = None) -> CorrectionResult:
        results = await self.correct([text], system_prompt)
        return results[0] if results else None

    async def _call_api(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Dict[str, Any]:
        import aiohttp
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": messages}
        if system:
            payload["system"] = system
        _debug(f"调用 API: {self.api_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API 错误: {response.status} - {text}")
                    return await response.json()
        except Exception as e:
            logger.error(f"调用纠错 API 失败: {e}")
            raise

    def _parse_response(self, response_data: Dict[str, Any], original_texts: List[str]) -> List[CorrectionResult]:
        try:
            choices = response_data.get("choices", [])
            if not choices:
                return [self._create_no_error_result(t) for t in original_texts]

            message = choices[0].get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None

            return self._parse_content(content, original_texts, reasoning_content)
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return [self._create_no_error_result(t) for t in original_texts]

    def _parse_content(self, content: str, original_texts: List[str], reasoning_content: Optional[str]) -> List[CorrectionResult]:
        think_match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        has_block_marker = bool(re.search(r'【块 \d+】\([^)]+\)', content_clean))

        if has_block_marker:
            parts = re.split(r'【块 \d+】\([^)]+\)', content_clean)
            if len(parts) >= 2:
                reasoning_part = parts[0].strip()
                corrected_lines = [c.strip() for c in parts[1].strip().split('\n') if c.strip()]
                error_type, explanation, full_reasoning = self._extract_meta(reasoning_part, reasoning_content, think_content)
                results = [
                    self._build_result(
                        original_texts[i],
                        corrected_lines[i] if i < len(corrected_lines) else original_texts[i],
                        error_type, explanation, full_reasoning
                    )
                    for i in range(len(original_texts))
                ]
                _debug(f"原文: {original_texts}")
                _debug(f"纠正后: {[r.corrected for r in results]}")
                return results
            return [self._create_no_error_result(t) for t in original_texts]

        elif '\n\n' in content_clean:
            sections = content_clean.split('\n\n', 1)
            reasoning_part = sections[0].strip()
            corrected_lines = [c.strip() for c in sections[1].strip().split('\n') if c.strip()] if len(sections) > 1 else []
            error_type, explanation, full_reasoning = self._extract_meta(reasoning_part, reasoning_content, think_content)
            results = [
                self._build_result(
                    original_texts[i],
                    corrected_lines[i] if i < len(corrected_lines) else original_texts[i],
                    error_type, explanation, full_reasoning
                )
                for i in range(len(original_texts))
            ]
            _debug(f"原文: {original_texts}")
            _debug(f"纠正后: {[r.corrected for r in results]}")
            return results

        elif content_clean.startswith('错误类型'):
            lines = content_clean.split('\n')
            reasoning_lines, corrected_lines, in_correction = [], [], False
            for line in lines:
                line = line.strip()
                if not line:
                    in_correction = True
                    continue
                (corrected_lines if in_correction else reasoning_lines).append(line)
            reasoning_part = '\n'.join(reasoning_lines)
            error_type, explanation, full_reasoning = self._extract_meta(reasoning_part, reasoning_content, think_content)
            results = [
                self._build_result(
                    original_texts[i],
                    corrected_lines[i] if i < len(corrected_lines) else original_texts[i],
                    error_type, explanation, full_reasoning
                )
                for i in range(len(original_texts))
            ]
            _debug(f"原文: {original_texts}")
            _debug(f"纠正后: {[r.corrected for r in results]}")
            return results

        else:
            full_reasoning = think_content or reasoning_content or ""
            error_type, explanation = "错别字", ""
            if think_content:
                error_type, explanation, _ = self._extract_meta(think_content, reasoning_content, "")

            lines = [l.strip() for l in content_clean.split('\n') if l.strip()]

            if len(lines) == len(original_texts):
                results = [
                    self._build_result(original_texts[i], lines[i], error_type, explanation, full_reasoning)
                    for i in range(len(original_texts))
                ]
            elif len(lines) > 0 and len(lines) <= len(original_texts) * 2:
                results = self._smart_align_results(lines, original_texts, error_type, explanation, full_reasoning)
            else:
                results = [
                    self._build_result(original_texts[i], lines[i] if i < len(lines) else original_texts[i], error_type, explanation, full_reasoning)
                    for i in range(len(original_texts))
                ]

            _debug(f"原文: {original_texts}")
            _debug(f"纠正后: {[r.corrected for r in results]}")
            return results

    def _smart_align_results(
        self,
        lines: List[str],
        original_texts: List[str],
        error_type: str,
        explanation: str,
        reasoning: str
    ) -> List[CorrectionResult]:
        """智能对齐纠正结果与原文"""
        results: List[Optional[CorrectionResult]] = [None] * len(original_texts)
        used_corrections: List[int] = []

        for i, line in enumerate(lines):
            for j, original in enumerate(original_texts):
                if results[j] is not None:
                    continue
                s_line = line.strip('。！？，,.!? \t\n\r')
                s_orig = original.strip('。！？，,.!? \t\n\r')
                if s_line == s_orig or s_orig in s_line or s_line in s_orig:
                    results[j] = self._build_result(original, line, error_type, explanation, reasoning)
                    used_corrections.append(i)
                    break

        for i, line in enumerate(lines):
            if i in used_corrections:
                continue
            best_match = -1
            best_score = 0
            for j, original in enumerate(original_texts):
                if results[j] is not None:
                    continue
                score = self._compute_similarity(line, original)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = j
            if best_match >= 0:
                results[best_match] = self._build_result(original_texts[best_match], line, error_type, explanation, reasoning)
                used_corrections.append(i)

        for j in range(len(original_texts)):
            if results[j] is None:
                results[j] = self._build_result(original_texts[j], original_texts[j], error_type, explanation, reasoning)

        return [r for r in results if r is not None]

    def _compute_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        s1_clean = s1.strip('。！？，,.!? \t\n\r')
        s2_clean = s2.strip('。！？，,.!? \t\n\r')
        if not s1_clean or not s2_clean:
            return 0.0
        if s1_clean == s2_clean:
            return 1.0
        n, m = len(s1_clean), len(s2_clean)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1_clean[i - 1] == s2_clean[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[n][m]
        union_len = n + m - lcs_len
        return lcs_len / union_len if union_len > 0 else 0.0

    def _extract_meta(self, reasoning_part: str, reasoning_content: Optional[str], think_content: str) -> Tuple[str, str, str]:
        error_type = "错别字"
        m = re.search(r'错误类型[:：]?\s*([^，,\n]+)', reasoning_part)
        if m:
            error_type = self._map_error_type(m.group(1).strip())

        explanation = reasoning_part
        match = re.search(r'修改[说明原因解释][:：]\s*(.+?)(?=错误类型[:：]|$)', reasoning_part, re.DOTALL)
        if match:
            explanation = match.group(1).strip()
        else:
            explanation = re.sub(r'错误类型[:：]?\s*[^，,\n]+\n?', '', explanation).strip()

        explanation = explanation.replace('\\n', '\n')

        parts = []
        if think_content:
            parts.append(think_content)
        if reasoning_content and reasoning_content not in (think_content or ""):
            parts.append(reasoning_content)
        if reasoning_part and reasoning_part not in (think_content or ""):
            parts.append(reasoning_part)
        full_reasoning = "\n".join(parts) if parts else reasoning_part

        return error_type, explanation, full_reasoning

    def _build_result(self, original: str, corrected: str, error_type: str, explanation: str, reasoning: str) -> CorrectionResult:
        if explanation:
            match = re.search(r'修改[说明原因解释][:：]\s*(.+?)(?=错误类型[:：]|$)', explanation, re.DOTALL)
            if match:
                explanation = match.group(1).strip().replace('\\n', '\n')

        # 验证原文和纠正是否匹配（长度差异不能太大，否则可能是对齐错误）
        # 如果 corrected 比 original 长很多（超过2倍），说明可能是对齐错误，跳过
        if len(corrected) > len(original) * 2:
            return self._create_no_error_result(original)

        # 如果 corrected 比 original 短很多（少于一半），可能是对齐错误
        if len(corrected) < len(original) * 0.5 and len(original) > 10:
            return self._create_no_error_result(original)

        spans = build_diff_spans(original, corrected)

        if not spans:
            if original != corrected:
                s_orig = original.strip('。！？，,.!? \t\n\r')
                s_corr = corrected.strip('。！？，,.!? \t\n\r')
                if s_orig == s_corr:
                    return self._create_no_error_result(original)
            return self._create_no_error_result(original)

        meaningful_spans = []
        for span in spans:
            if _is_meaningful_diff(span.original_text, span.corrected_text):
                meaningful_spans.append(span)

        if not meaningful_spans:
            return self._create_no_error_result(original)

        return CorrectionResult(
            original=original,
            corrected=corrected,
            error_type=error_type,
            explanation=explanation,
            reasoning=reasoning,
            diff_spans=meaningful_spans,
        )

    def _map_error_type(self, type_text: str) -> str:
        mapping = {
            "错别字": "错别字",
            "词语搭配错误": "词语搭配错误",
            "词性错误": "词性错误",
            "语序错误": "语序错误",
            "成分残缺": "成分残缺",
            "成分赘余": "成分赘余",
            "关联词使用错误": "关联词使用错误",
            "指代不明": "指代不明",
            "语义逻辑不通": "语义逻辑不通"
        }
        for key, value in mapping.items():
            if key in type_text:
                return value
        return "错别字"

    def _create_no_error_result(self, text: str) -> CorrectionResult:
        return CorrectionResult(original=text, corrected=text, error_type="", explanation="", reasoning="", diff_spans=[])

    def match_result_with_original(self, results: List[CorrectionResult], original_texts: List[str]) -> List[CorrectionResult]:
        """确保每个结果都有正确的 original 文本"""
        matched = []

        for result in results:
            # 如果已经有 original，直接使用
            if result.original:
                # 验证 original 是否在 original_texts 中
                if result.original not in original_texts:
                    # 尝试找到最匹配的 original
                    best_match = None
                    best_score = 0
                    for orig in original_texts:
                        score = self._compute_similarity(result.original, orig)
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = orig
                    if best_match:
                        result.original = best_match
                        result.diff_spans = build_diff_spans(best_match, result.corrected)
                    else:
                        # 无法匹配，跳过
                        continue
                matched.append(result)
                continue

            # 尝试匹配 corrected 和 original
            s_corr, _ = _strip_edge_punct(result.corrected)
            best_match = None
            best_score = 0

            for original in original_texts:
                if original in [r.original for r in matched]:
                    continue
                s_orig, _ = _strip_edge_punct(original)

                # 检查是否完全匹配或包含
                if s_orig == s_corr or s_orig in s_corr or s_corr in s_orig:
                    best_match = original
                    best_score = 1.0
                    break

                # 计算相似度
                score = self._compute_similarity(s_corr, s_orig)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = original

            if best_match:
                result.original = best_match
                result.diff_spans = build_diff_spans(best_match, result.corrected)
                matched.append(result)
            # 如果没有匹配，跳过这个结果

        # 补充缺失的 original
        used_originals = set(r.original for r in matched)
        for original in original_texts:
            if original not in used_originals:
                matched.append(self._create_no_error_result(original))

        return matched


class ChineseCorrectorSyncAdapter:
    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: int = 120):
        import requests
        self.requests = requests
        self.api_url = api_url.rstrip("/")
        if "/chat/completions" not in self.api_url:
            self.api_url = f"{self.api_url}/chat/completions"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.async_adapter = None

    def _get_async_adapter(self) -> ChineseCorrectorAdapter:
        if self.async_adapter is None:
            self.async_adapter = ChineseCorrectorAdapter(
                api_url=self.api_url, api_key=self.api_key,
                model_name=self.model_name, timeout=self.timeout
            )
        return self.async_adapter

    def correct(self, texts: List[str], system_prompt: Optional[str] = None) -> List[CorrectionResult]:
        if not texts:
            return []

        _debug(f"原文: {texts}")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        system_content = system_prompt or "假如你是一名专业的纠错专家，请分析输入句子的语法错误类型和修改原因，并只输出纠正后的语句，错误类型如下：错别字、词语搭配错误、词性错误、语序错误、成分残缺、成分赘余、关联词使用错误、指代不明、语义逻辑不通、无误。"
        user_content = "\n".join([text.strip() for text in texts if text.strip()])
        messages = [{"role": "user", "content": user_content}]
        payload = {"model": self.model_name, "system": system_content, "messages": messages}
        logger.info(f"[ChineseCorrectorSync] 发送请求")

        try:
            response = self.requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return [self._create_result(t, t, "", "", "") for t in texts]
            message = choices[0].get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None

            adapter = self._get_async_adapter()
            results = adapter._parse_content(content, texts, reasoning_content)
            results = adapter.match_result_with_original(results, texts)

            while len(results) < len(texts):
                results.append(adapter._create_no_error_result(texts[len(results)]))

            _debug(f"纠正后: {[r.corrected for r in results]}")
            return results[:len(texts)]
        except Exception as e:
            logger.error(f"[ChineseCorrectorSync] 纠错失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"[ChineseCorrectorSync] 响应内容: {e.response.text[:500]}")
            return [self._create_result(t, t, "", str(e), "") for t in texts]

    def correct_single(self, text: str, system_prompt: Optional[str] = None) -> CorrectionResult:
        results = self.correct([text], system_prompt)
        return results[0] if results else None

    def _create_result(self, original: str, corrected: str, error_type: str, explanation: str, reasoning: str) -> CorrectionResult:
        return CorrectionResult(
            original=original, corrected=corrected,
            error_type=error_type, explanation=explanation, reasoning=reasoning,
            diff_spans=build_diff_spans(original, corrected),
        )
