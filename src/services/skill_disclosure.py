"""分类感知的 Skills 渐进式披露（progressive disclosure）。

参照 hermes 的执行逻辑，把 skills 的暴露拆成两层：

1. 菜单层（tier-1）：system prompt 里只放「分类清单 + 各分类 DESCRIPTION.md 摘要」，
   不再把每个 skill 的全文/描述全量常驻。模型调用 ``skills_list(category)`` 才按需
   拿到该分类下 skill 的 name+description。
2. 全文层（tier-2）：模型确定要用某个 skill 后，调用 ``skill_view(name)`` 加载其
   SKILL.md 全文。

目录约定（与 hermes 一致）：
    skills/<category>/<skill>/SKILL.md   -> category = <category>
    skills/<skill>/SKILL.md              -> 无分类（顶层 skill）
每个 <category>/ 下可放一个 DESCRIPTION.md（frontmatter 里的 description 作为分类摘要）。

多租户：roots 按优先级排序（内置在前、租户在后），同名 skill 后者覆盖前者。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_LENGTH = 1024
# 扫描时跳过的目录（隐藏目录、hub 元数据、脚本/引用子目录里不会再嵌 skill）
_EXCLUDED_DIR_PARTS = {".hub", "__pycache__", ".git"}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(content: str) -> Dict:
    """解析 SKILL.md / DESCRIPTION.md 顶部的 YAML frontmatter。

    yaml 解析失败时（如 description 引号未闭合等不规范写法）回退到逐行提取，
    保证至少能拿到可用的 name/description，对用户上传的不规范 skill 也健壮。
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}
    block = match.group(1)
    try:
        data = yaml.safe_load(block)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError as e:
        logger.warning("frontmatter YAML 解析失败，回退逐行提取: %s", e)
    return _parse_frontmatter_lenient(block)


def _parse_frontmatter_lenient(block: str) -> Dict:
    """逐行提取顶层 key: value，容忍未闭合引号与缩进续行。"""
    result: Dict[str, str] = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if current_key is None:
            return
        val = " ".join(s.strip() for s in buf).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:  # 未闭合引号：去掉首个引号
            val = val.lstrip("\"'")
        result[current_key] = val.strip()

    for line in block.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s?(.*)$", line)
        if m:
            flush()
            current_key = m.group(1)
            rest = m.group(2).strip()
            buf = [] if rest in (">", "|", ">-", "|-") else [rest]
        elif current_key is not None and (line.startswith((" ", "\t")) or line.strip()):
            buf.append(line)
    flush()
    return result


@dataclass
class SkillEntry:
    name: str
    description: str
    category: Optional[str]
    skill_dir: Path
    skill_md: Path
    source: str  # "builtin" | "user"


class SkillRegistry:
    """递归扫描分类化的 skill 目录，提供菜单/全文按需查询。"""

    def __init__(self, roots: List[Tuple[Path, str]]):
        """roots: [(目录, source 标签)]，按优先级升序（后者覆盖前者）。"""
        self._roots = [(Path(p), src) for p, src in roots]
        self._skills: Dict[str, SkillEntry] = {}
        self._categories: Dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        for root, source in self._roots:
            if not root.exists():
                continue
            self._scan_categories(root)
            for skill_md in root.rglob("SKILL.md"):
                if any(part in _EXCLUDED_DIR_PARTS for part in skill_md.parts):
                    continue
                entry = self._build_entry(skill_md, root, source)
                if entry:
                    # 后者覆盖前者（租户覆盖内置）
                    self._skills[entry.name] = entry

    def _scan_categories(self, root: Path) -> None:
        for child in root.iterdir():
            if not child.is_dir() or child.name in _EXCLUDED_DIR_PARTS:
                continue
            desc_md = child / "DESCRIPTION.md"
            if desc_md.exists():
                fm = _parse_frontmatter(desc_md.read_text(encoding="utf-8"))
                desc = str(fm.get("description", "")).strip()
                if desc:
                    self._categories[child.name] = desc

    def _build_entry(self, skill_md: Path, root: Path, source: str) -> Optional[SkillEntry]:
        try:
            content = skill_md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.debug("跳过无法读取的 skill %s: %s", skill_md, e)
            return None

        fm = _parse_frontmatter(content)
        skill_dir = skill_md.parent
        name = str(fm.get("name", skill_dir.name)).strip() or skill_dir.name

        description = str(fm.get("description", "")).strip()
        if not description:
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "-")):
                    description = line
                    break
        description = " ".join(description.split())
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

        category = self._category_from_path(skill_md, root)
        return SkillEntry(
            name=name,
            description=description,
            category=category,
            skill_dir=skill_dir,
            skill_md=skill_md,
            source=source,
        )

    @staticmethod
    def _category_from_path(skill_md: Path, root: Path) -> Optional[str]:
        try:
            parts = skill_md.relative_to(root).parts
        except ValueError:
            return None
        # category/skill/SKILL.md -> 3 段；顶层 skill/SKILL.md -> 2 段（无分类）
        return parts[0] if len(parts) >= 3 else None

    # ---- 查询接口 ----------------------------------------------------------

    def categories(self) -> Dict[str, str]:
        """返回 {分类: 摘要}，含没有 DESCRIPTION.md 的分类（摘要为空）。"""
        result = dict(self._categories)
        for entry in self._skills.values():
            if entry.category and entry.category not in result:
                result[entry.category] = ""
        return result

    def list(self, category: Optional[str] = None) -> List[SkillEntry]:
        entries = list(self._skills.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: (e.category or "", e.name))

    def resolve(self, name: str) -> Optional[SkillEntry]:
        """按 name 或 'category/name' 解析单个 skill。"""
        name = name.strip()
        if "/" in name:
            cat, _, bare = name.partition("/")
            for e in self._skills.values():
                if e.name == bare and e.category == cat:
                    return e
            return None
        return self._skills.get(name)


# ---------------------------------------------------------------------------
# 工具与 system-prompt 概览构建
# ---------------------------------------------------------------------------


class _SkillsListArgs(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="可选的分类名，用于只列出该分类下的 skill（如 document、design、research）。留空则列出全部。",
    )


class _SkillViewArgs(BaseModel):
    name: str = Field(
        description="要查看的 skill 名称；同名冲突时用 '分类/名称' 消歧（如 research/rag-query）。",
    )


_SKILLS_LIST_DESC = (
    "列出可用 skill 的菜单（仅 name + description + category，不含全文）。"
    "当用户的需求可能匹配某类专长时先调用它；可传 category 收窄范围。"
    "看到合适的 skill 后，再用 skill_view 加载其完整说明。"
)

_SKILL_VIEW_DESC = (
    "加载指定 skill 的完整 SKILL.md 说明（含执行步骤、脚本路径等）。"
    "在 skills_list 选定 skill 后调用，然后严格按其说明执行。"
)


def _make_skills_list_tool(registry: SkillRegistry) -> StructuredTool:
    def _run(category: Optional[str] = None) -> str:
        entries = registry.list(category)
        payload = {
            "skills": [
                {"name": e.name, "description": e.description, "category": e.category}
                for e in entries
            ],
            "categories": sorted(registry.categories().keys()),
            "count": len(entries),
            "hint": "用 skill_view(name) 查看完整说明",
        }
        return json.dumps(payload, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_run,
        name="skills_list",
        description=_SKILLS_LIST_DESC,
        args_schema=_SkillsListArgs,
    )


def _make_skill_view_tool(registry: SkillRegistry) -> StructuredTool:
    def _run(name: str) -> str:
        entry = registry.resolve(name)
        if not entry:
            avail = ", ".join(e.name for e in registry.list())
            return json.dumps(
                {"error": f"未找到 skill '{name}'", "available": avail},
                ensure_ascii=False,
            )
        try:
            body = entry.skill_md.read_text(encoding="utf-8")
        except OSError as e:
            return json.dumps({"error": f"读取 SKILL.md 失败: {e}"}, ensure_ascii=False)
        # 头部给出 skill 目录绝对路径，便于按说明执行其中的脚本
        header = f"<!-- skill_dir: {entry.skill_dir} -->\n"
        return header + body

    return StructuredTool.from_function(
        func=_run,
        name="skill_view",
        description=_SKILL_VIEW_DESC,
        args_schema=_SkillViewArgs,
    )


def build_skill_disclosure(
    builtin_dir: Path,
    tenant_dir: Optional[Path] = None,
) -> Tuple[List[StructuredTool], str]:
    """构建渐进披露：返回 (工具列表, system-prompt 概览文本)。

    工具：skills_list / skill_view。
    概览：分类清单 + 各分类 DESCRIPTION.md 摘要 + 使用说明，注入 system prompt。
    """
    roots: List[Tuple[Path, str]] = [(Path(builtin_dir), "builtin")]
    if tenant_dir is not None:
        roots.append((Path(tenant_dir), "user"))

    registry = SkillRegistry(roots)
    tools = [_make_skills_list_tool(registry), _make_skill_view_tool(registry)]
    overview = _build_overview(registry)
    return tools, overview


def _build_overview(registry: SkillRegistry) -> str:
    cats = registry.categories()
    lines = [
        "## 技能系统（渐进式披露）",
        "",
        "你有一个分类化的技能库。下面只列出**分类**及其用途——不含具体技能清单。",
        "需要时按两步展开：",
        "1. 调用 `skills_list(category)` 拿到该分类下的技能（name+description）；",
        "2. 选定后调用 `skill_view(name)` 加载完整说明，再严格执行。",
        "",
        "**可用分类：**",
    ]
    if cats:
        for cat in sorted(cats):
            summary = cats[cat] or "（无摘要）"
            lines.append(f"- **{cat}**：{summary}")
    else:
        lines.append("- （暂无分类技能）")

    uncategorized = [e for e in registry.list() if not e.category]
    if uncategorized:
        lines.append("")
        lines.append("**未分类技能**（可直接 skill_view 查看）：")
        for e in uncategorized:
            lines.append(f"- **{e.name}**：{e.description}")

    lines.append("")
    lines.append("闲聊或简单问题直接回答，无需调用以上工具。")
    return "\n".join(lines)
