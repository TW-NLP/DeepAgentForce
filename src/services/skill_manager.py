"""
Skill 管理服务
负责动态加载、卸载、验证前端上传的 Skill
支持 Skill 的即插即用

多租户设计：
- 内置 Skills（src/services/skills/）：所有用户可见
- 用户自定义 Skills（data/user_skills/{tenant_uuid}/）：按租户隔离
"""

import logging
import shutil
import uuid
import json
import tempfile
import zipfile
import yaml  # 🆕 用于正确解析 YAML frontmatter
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 内置 Skill 标识符（不允许删除）
BUILTIN_SKILL_IDS = {'pdf-processing', 'rag-query', 'web-search'}


class SkillManager:
    """
    Skill 管理器 - 支持前端动态上传 Skill

    多租户隔离策略：
    1. 内置 Skills：从 src/services/skills/ 加载，所有用户可见
    2. 用户自定义 Skills：从 data/user_skills/{tenant_uuid}/ 加载，按租户隔离

    当需要获取用户的完整 Skill 列表时，会合并两类 Skills，
    并标注每个 Skill 的来源（builtin/user）和所有者。
    """

    def __init__(self, builtin_skills_dir: Path, user_skills_base_dir: Path):
        """
        初始化 SkillManager

        Args:
            builtin_skills_dir: 内置 Skills 目录（src/services/skills/，只读模板）
            user_skills_base_dir: 用户 Skills 根目录（data/skill/）
        """
        self.builtin_skills_dir = Path(builtin_skills_dir)
        self.user_skills_base_dir = Path(user_skills_base_dir)
        self.user_skills_base_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存：{skill_id: skill_info}
        self._skills_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_tenant_uuid: Optional[str] = None

    def _get_tenant_skill_dir(self, tenant_uuid: Optional[str]) -> Path:
        """获取指定租户的 Skills 目录"""
        if tenant_uuid is None:
            return self.user_skills_base_dir
        tenant_dir = self.user_skills_base_dir / tenant_uuid
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir

    def initialize_tenant_skills(self, tenant_uuid: str) -> None:
        """🆕 首次初始化租户的 Skills：创建用户目录结构"""
        tenant_dir = self._get_tenant_skill_dir(tenant_uuid)
        if tenant_dir.exists() and any(tenant_dir.iterdir()):
            return  # 已初始化过，跳过

        logger.info(f"初始化租户 {tenant_uuid} 的 Skills 目录...")
        # 只创建目录，不复制内置 Skills
        # 内置 Skills 通过 workspace: src/services/skills:data/skill/{tenant_uuid} 访问

    def _resolve_skill_path(self, skill_id: str, tenant_uuid: Optional[str] = None) -> Optional[Path]:
        """
        解析 Skill 的实际路径

        查找顺序：
        1. 用户自定义目录（如果指定了 tenant_uuid）
        2. 内置 Skills 目录

        Args:
            skill_id: Skill 标识符
            tenant_uuid: 租户UUID（可选）

        Returns:
            Skill 目录路径，如果不存在返回 None
        """
        # 1. 优先查找用户自定义目录
        if tenant_uuid is not None:
            tenant_dir = self._get_tenant_skill_dir(tenant_uuid)
            tenant_skill_path = tenant_dir / skill_id
            if tenant_skill_path.exists():
                return tenant_skill_path

        # 2. 查找内置 Skills 目录
        builtin_path = self.builtin_skills_dir / skill_id
        if builtin_path.exists():
            return builtin_path

        return None

    def _is_builtin_skill(self, skill_id: str) -> bool:
        """判断是否为内置 Skill"""
        return skill_id in BUILTIN_SKILL_IDS

    def list_skills(self, tenant_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出指定租户的 Skills 列表

        返回：
        - 内置 Skills（所有用户可见，来自 src/services/skills/）
        - 租户自己的 Skills（来自 src/services/skills/_user_skills/{tenant_uuid}/）

        Args:
            tenant_uuid: 租户UUID（用于获取租户自己的 Skills）

        Returns:
            Skills 列表，按名称排序
        """
        skills = []

        # 1. 添加内置 Skills（排除用户目录）
        if self.builtin_skills_dir.exists():
            for skill_path in self.builtin_skills_dir.iterdir():
                if not skill_path.is_dir():
                    continue
                # 跳过用户 Skills 目录
                if skill_path.name == '_user_skills':
                    continue
                skill_info = self._parse_skill_info(skill_path)
                if skill_info:
                    skill_info['source'] = 'builtin'
                    skill_info['owner_id'] = None  # 内置 Skill 无所有者
                    skill_info['is_deletable'] = False
                    skills.append(skill_info)

        # 2. 添加指定租户的 Skills
        if tenant_uuid is not None:
            tenant_dir = self._get_tenant_skill_dir(tenant_uuid)
            if tenant_dir.exists():
                for skill_path in tenant_dir.iterdir():
                    if not skill_path.is_dir():
                        continue
                    skill_info = self._parse_skill_info(skill_path)
                    if skill_info:
                        skill_info['source'] = 'user'
                        skill_info['owner_id'] = tenant_uuid
                        skill_info['is_deletable'] = True
                        skills.append(skill_info)

        # 按名称排序
        skills.sort(key=lambda x: (x.get('source', '') != 'builtin', x.get('name', '')))
        return skills

    def _collect_scripts(self, skill_path: Path) -> Dict[str, str]:
        """收集 Skill 中 scripts 目录下的 Python 脚本"""
        scripts: Dict[str, str] = {}
        scripts_dir = skill_path / "scripts"
        if not scripts_dir.exists():
            return scripts

        for script_file in scripts_dir.iterdir():
            if script_file.is_file() and script_file.suffix == '.py':
                scripts[script_file.name] = script_file.read_text(encoding='utf-8')

        return scripts

    def _parse_skill_info(self, skill_path: Path) -> Optional[Dict[str, Any]]:
        """解析 Skill 目录，提取元信息"""
        skill_md = skill_path / "SKILL.md"

        if not skill_md.exists():
            return None

        try:
            # 读取 SKILL.md 解析 frontmatter
            content = skill_md.read_text(encoding='utf-8')
            metadata = self._extract_frontmatter(content)

            # 获取 scripts 目录信息
            scripts = []
            script_names = []
            scripts_dir = skill_path / "scripts"
            if scripts_dir.exists():
                for script_file in scripts_dir.iterdir():
                    if script_file.is_file() and script_file.suffix == '.py':
                        script_names.append(script_file.name)
                        scripts.append({
                            'name': script_file.stem,
                            'file': script_file.name,
                            'path': str(script_file.relative_to(skill_path.parent))
                        })

            # 统计信息
            created_at = datetime.fromtimestamp(skill_path.stat().st_ctime).isoformat()
            modified_at = datetime.fromtimestamp(skill_path.stat().st_mtime).isoformat()
            size_bytes = 0
            file_count = 0
            for item in skill_path.rglob('*'):
                if item.is_file():
                    file_count += 1
                    try:
                        size_bytes += item.stat().st_size
                    except OSError:
                        continue

            summary = metadata.get('description', '')
            if not summary:
                body_lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith('---')]
                summary = body_lines[0][:120] if body_lines else ''

            return {
                'id': skill_path.name,
                'name': metadata.get('name', skill_path.name),
                'description': metadata.get('description', ''),
                'version': metadata.get('version', '1.0.0'),
                'author': metadata.get('author', 'Unknown'),
                'tags': metadata.get('tags', []),
                'path': str(skill_path),
                'scripts': scripts,
                'script_count': len(scripts),
                'script_names': script_names,
                'size_bytes': size_bytes,
                'file_count': file_count,
                'summary': summary,
                'is_builtin': skill_path.parent == self.builtin_skills_dir or skill_path.name in BUILTIN_SKILL_IDS,
                'is_custom': skill_path.parent != self.builtin_skills_dir and skill_path.name not in BUILTIN_SKILL_IDS,
                'created_at': created_at,
                'modified_at': modified_at
            }
        except Exception as e:
            logger.error(f"解析 Skill 失败 {skill_path}: {e}")
            return None

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """提取 YAML frontmatter，使用 PyYAML 正确解析"""
        metadata = {}

        if content.startswith('---'):
            try:
                lines = content.split('\n')
                # 找到 frontmatter 的结束位置
                end_idx = 1
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '---':
                        end_idx = i
                        break

                # 提取 frontmatter 部分
                frontmatter_content = '\n'.join(lines[1:end_idx])

                # 使用 PyYAML 解析
                parsed = yaml.safe_load(frontmatter_content)
                if parsed and isinstance(parsed, dict):
                    metadata = parsed

            except yaml.YAMLError as e:
                logger.warning(f"YAML 解析失败，使用正则 fallback: {e}")
                # 如果 PyYAML 解析失败，回退到简单的正则解析
                metadata = self._extract_frontmatter_regex(content)

        return metadata

    def _extract_frontmatter_regex(self, content: str) -> Dict[str, Any]:
        """正则方式提取 frontmatter（fallback）"""
        metadata = {}

        if content.startswith('---'):
            lines = content.split('\n')
            in_frontmatter = False
            key_buffer = []
            value_buffer = []
            current_key = None

            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    if not in_frontmatter:
                        in_frontmatter = True
                        continue
                    else:
                        break

                if in_frontmatter and ':' in line:
                    # 处理多行值
                    if line.startswith(' ') or line.startswith('\t'):
                        if current_key:
                            value_buffer.append(line.strip())
                    else:
                        # 保存前一个 key-value
                        if current_key:
                            key = current_key.strip()
                            value = ':'.join(value_buffer).strip()
                            # 处理列表
                            if value.startswith('[') and value.endswith(']'):
                                value = [v.strip().strip('"\'') for v in value[1:-1].split(',')]
                            metadata[key] = value

                        # 开始新的 key
                        idx = line.index(':')
                        current_key = line[:idx]
                        value_buffer = [line[idx+1:].strip()]

            # 保存最后一个 key-value
            if current_key:
                key = current_key.strip()
                value = ':'.join(value_buffer).strip()
                if value.startswith('[') and value.endswith(']'):
                    value = [v.strip().strip('"\'') for v in value[1:-1].split(',')]
                metadata[key] = value

        return metadata

    def get_skill(self, skill_id: str, tenant_uuid: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        获取指定 Skill 的详细信息

        Args:
            skill_id: Skill 标识符
            tenant_uuid: 用户ID（可选）

        Returns:
            Skill 信息，如果不存在返回 None
        """
        skill_path = self._resolve_skill_path(skill_id, tenant_uuid)
        if not skill_path:
            return None

        skill_info = self._parse_skill_info(skill_path)
        if skill_info:
            skill_info['source'] = 'builtin' if self._is_builtin_skill(skill_id) else 'user'
            skill_info['owner_id'] = None if self._is_builtin_skill(skill_id) else tenant_uuid
            skill_info['is_deletable'] = not self._is_builtin_skill(skill_id)
        return skill_info

    def get_skill_content(self, skill_id: str, tenant_uuid: Optional[int] = None) -> Optional[Dict[str, str]]:
        """
        获取 Skill 文件内容 (SKILL.md 和 scripts)

        Args:
            skill_id: Skill 标识符
            tenant_uuid: 用户ID（可选）

        Returns:
            包含 SKILL.md 和 scripts 的字典
        """
        skill_path = self._resolve_skill_path(skill_id, tenant_uuid)
        if not skill_path:
            return None

        result = {}

        # 读取 SKILL.md
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            result['SKILL.md'] = skill_md.read_text(encoding='utf-8')

        # 读取 scripts
        result['scripts'] = self._collect_scripts(skill_path)

        return result

    def validate_skill(self, skill_md_content: str, scripts: Dict[str, str]) -> Dict[str, Any]:
        """验证 Skill 规范的合法性"""
        errors = []
        warnings = []

        # 1. 验证 SKILL.md
        if not skill_md_content.strip():
            errors.append("SKILL.md 内容不能为空")
        else:
            metadata = self._extract_frontmatter(skill_md_content)

            if not metadata.get('name'):
                errors.append("SKILL.md 缺少 name 字段")

            if not metadata.get('description'):
                warnings.append("建议添加 description 字段")

        # 2. 验证 scripts
        if not scripts:
            warnings.append("建议至少提供一个脚本")
        else:
            for script_name, content in scripts.items():
                if not content.strip():
                    errors.append(f"脚本 {script_name} 内容为空")
                # 检查是否是有效的 Python 文件
                if script_name.endswith('.py'):
                    try:
                        compile(content, script_name, 'exec')
                    except SyntaxError as e:
                        warnings.append(f"脚本 {script_name} 存在语法问题: {e}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def install_skill(
        self,
        skill_name: str,
        skill_md_content: str,
        scripts: Dict[str, str],
        tenant_uuid: int,
        force: bool = False,
        source_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        安装/更新用户自定义 Skill

        用户创建的 Skills 会存储到 data/user_skills/{tenant_uuid}/ 目录下，
        实现用户级别的数据隔离。

        Args:
            skill_name: Skill 名称
            skill_md_content: SKILL.md 内容
            scripts: 脚本字典
            tenant_uuid: 用户ID（用于确定存储位置）
            force: 是否强制覆盖

        Returns:
            安装结果字典
        """
        # 验证合法性
        validation = self.validate_skill(skill_md_content, scripts)

        if not validation['valid']:
            return {
                'success': False,
                'message': '验证失败',
                'errors': validation['errors']
            }

        # 生成 Skill ID
        skill_id = self._generate_skill_id(skill_name)

        # 内置 Skills 不允许覆盖
        if self._is_builtin_skill(skill_id):
            return {
                'success': False,
                'message': f"无法覆盖内置 Skill '{skill_name}'"
            }

        # 用户 Skills 存储到用户专属目录
        user_dir = self._get_tenant_skill_dir(tenant_uuid)
        skill_path = user_dir / skill_id

        # 检查是否已存在
        if skill_path.exists() and not force:
            return {
                'success': False,
                'message': f"Skill '{skill_name}' 已存在，请使用更新接口"
            }

        try:
            # 创建目录结构
            if skill_path.exists():
                shutil.rmtree(skill_path)

            if source_path is not None:
                shutil.copytree(source_path, skill_path)
            else:
                skill_path.mkdir(parents=True, exist_ok=True)
                scripts_dir = skill_path / "scripts"
                scripts_dir.mkdir(parents=True, exist_ok=True)

                # 写入 SKILL.md
                (skill_path / "SKILL.md").write_text(skill_md_content, encoding='utf-8')

                # 写入 scripts
                for script_name, content in scripts.items():
                    script_path = scripts_dir / script_name
                    script_path.parent.mkdir(parents=True, exist_ok=True)
                    script_path.write_text(content, encoding='utf-8')

            # 清除缓存
            self._skills_cache.clear()

            return {
                'success': True,
                'message': f"Skill '{skill_name}' 安装成功",
                'skill_id': skill_id,
                'warnings': validation['warnings']
            }

        except Exception as e:
            logger.error(f"安装 Skill 失败: {e}")
            # 清理失败的文件
            if skill_path.exists():
                shutil.rmtree(skill_path)

            return {
                'success': False,
                'message': f"安装失败: {str(e)}"
            }

    def _generate_skill_id(self, skill_name: str) -> str:
        """从 Skill 名称生成 ID"""
        import re
        skill_id = skill_name.lower()
        skill_id = re.sub(r'[^a-z0-9]+', '-', skill_id)
        skill_id = skill_id.strip('-')
        return skill_id

    def uninstall_skill(self, skill_id: str, tenant_uuid: Optional[int] = None) -> Dict[str, Any]:
        """
        卸载 Skill

        Args:
            skill_id: Skill 标识符
            tenant_uuid: 用户ID（用于验证权限）

        Returns:
            卸载结果字典
        """
        # 1. 内置 Skills 不允许删除
        if self._is_builtin_skill(skill_id):
            return {
                'success': False,
                'message': "内置 Skill 不能删除"
            }

        # 2. 用户只能删除自己的 Skills
        skill_path = self._resolve_skill_path(skill_id, tenant_uuid)
        if not skill_path:
            return {
                'success': False,
                'message': f"Skill '{skill_id}' 不存在或无权限访问"
            }

        # 3. 验证用户是否有权限删除（skill 必须在用户目录下）
        if tenant_uuid is not None:
            expected_user_dir = self._get_tenant_skill_dir(tenant_uuid)
            if not str(skill_path).startswith(str(expected_user_dir)):
                return {
                    'success': False,
                    'message': "无权限删除此 Skill"
                }

        try:
            shutil.rmtree(skill_path)
            self._skills_cache.clear()

            return {
                'success': True,
                'message': f"Skill '{skill_id}' 已卸载"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"卸载失败: {str(e)}"
            }

    def _extract_zip_package(self, archive_path: Path, extract_root: Path) -> None:
        """安全解压 ZIP 包到临时目录"""
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or '..' in member_path.parts:
                    raise ValueError("压缩包包含非法路径")

                target_path = extract_root / member_path
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)

    def _locate_skill_package_root(self, extract_root: Path) -> Optional[Path]:
        """定位压缩包中的 Skill 根目录"""
        skill_md_files = [p for p in extract_root.rglob('SKILL.md') if p.is_file()]
        if not skill_md_files:
            return None

        roots = {p.parent.resolve() for p in skill_md_files}
        if len(roots) != 1:
            raise ValueError("压缩包中只能包含一个 Skill 目录")

        return skill_md_files[0].parent

    def _normalize_skill_md(self, skill_md_content: str, skill_name: str) -> str:
        """确保 SKILL.md 至少包含 name 字段"""
        metadata = self._extract_frontmatter(skill_md_content)
        if metadata.get('name'):
            return skill_md_content

        body = skill_md_content
        if skill_md_content.startswith('---'):
            lines = skill_md_content.splitlines()
            end_idx = None
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    end_idx = i
                    break

            if end_idx is not None:
                body = '\n'.join(lines[end_idx + 1:]).lstrip('\n')

        metadata['name'] = skill_name
        frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{frontmatter}\n---\n\n{body}".rstrip() + "\n"

    def import_skill_package(
        self,
        archive_path: Path,
        tenant_uuid: int,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        从 ZIP 压缩包导入 Skill
        """
        if not archive_path.exists():
            return {
                'success': False,
                'message': '压缩包不存在'
            }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                extract_root = Path(tmpdir) / 'skill_package'
                extract_root.mkdir(parents=True, exist_ok=True)

                self._extract_zip_package(archive_path, extract_root)
                skill_root = self._locate_skill_package_root(extract_root)
                if not skill_root:
                    return {
                        'success': False,
                        'message': '压缩包中未找到 SKILL.md'
                    }

                skill_md_path = skill_root / 'SKILL.md'
                if not skill_md_path.exists():
                    return {
                        'success': False,
                        'message': '压缩包中未找到 SKILL.md'
                    }

                skill_md_content = skill_md_path.read_text(encoding='utf-8')
                metadata = self._extract_frontmatter(skill_md_content)
                skill_name = (metadata.get('name') or skill_root.name or archive_path.stem).strip()
                skill_md_content = self._normalize_skill_md(skill_md_content, skill_name)

                scripts = self._collect_scripts(skill_root)
                validation = self.validate_skill(skill_md_content, scripts)
                if not validation['valid']:
                    return {
                        'success': False,
                        'message': '验证失败',
                        'errors': validation['errors'],
                        'warnings': validation['warnings']
                    }

                result = self.install_skill(
                    skill_name=skill_name,
                    skill_md_content=skill_md_content,
                    scripts=scripts,
                    tenant_uuid=tenant_uuid,
                    force=force,
                    source_path=skill_root
                )

                if result.get('success'):
                    result['message'] = f"Skill '{skill_name}' 导入成功"
                    existing_warnings = result.get('warnings', [])
                    result['warnings'] = list(dict.fromkeys(existing_warnings + validation['warnings']))
                return result
        except zipfile.BadZipFile:
            return {
                'success': False,
                'message': '压缩包格式无效，请上传 ZIP 文件'
            }
        except ValueError as e:
            return {
                'success': False,
                'message': str(e),
                'errors': [str(e)]
            }
        except Exception as e:
            logger.error(f"导入 Skill 包失败: {e}")
            return {
                'success': False,
                'message': f"导入失败: {str(e)}",
                'errors': [str(e)]
            }

    def export_skill(self, skill_id: str, tenant_uuid: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        导出 Skill 为可分享的格式

        Args:
            skill_id: Skill 标识符
            tenant_uuid: 用户ID（可选）

        Returns:
            导出数据字典
        """
        skill_path = self._resolve_skill_path(skill_id, tenant_uuid)
        if not skill_path:
            return None

        content = self.get_skill_content(skill_id, tenant_uuid)
        if not content:
            return None

        # 构建导出结构
        skill_md = content.pop('SKILL.md', '')
        scripts = content.pop('scripts', {})

        return {
            'skill_md': skill_md,
            'scripts': scripts,
            'exported_at': datetime.now().isoformat()
        }

    def get_skill_template(self) -> Dict[str, Any]:
        """获取 Skill 模板"""
        return {
            'skill_md': '''---
name: my-custom-skill
description: A custom skill for specific tasks
version: 1.0.0
author: Your Name
tags:
  - custom
  - utility
---

# My Custom Skill

## When to use this skill

Use this skill when the user wants to:
- Do something specific
- Solve a particular problem

## Available Scripts

### 1. Main Script: `main.py`

**When to use**: Always

**Usage**:
```bash
python src/services/skills/my-custom-skill/scripts/main.py --param1 value1
```

**Parameters**:
- `--param1`: Description of param1

**Example**:
```bash
python src/services/skills/my-custom-skill/scripts/main.py --param1 hello
```
''',
            'scripts': {
                'main.py': '''#!/usr/bin/env python3
"""
Custom Skill Main Script
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Custom skill script")
    parser.add_argument('--param1', type=str, default='default', help='Parameter 1')
    args = parser.parse_args()
    
    # Your logic here
    print(f"Hello, {args.param1}!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
'''
            }
        }
