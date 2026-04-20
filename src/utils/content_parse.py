from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from typing import List, Optional
import io
from pypdf import PdfReader
from docx import Document


def _docx_paragraph_to_text(para) -> str:
    """
    尽量保留 Word 段落的视觉缩进。

    docx 解析拿不到完整版式，但可以把常见的左缩进 / 首行缩进近似为前导空格，
    这样导入后的文本在编辑区里会更接近原文。
    """
    text = para.text or ""
    if not text:
        return ""

    paragraph_format = getattr(para, "paragraph_format", None)
    indent_pt = 0.0

    if paragraph_format is not None:
        left_indent = getattr(paragraph_format, "left_indent", None)
        first_line_indent = getattr(paragraph_format, "first_line_indent", None)

        if left_indent is not None and getattr(left_indent, "pt", None):
            indent_pt += max(left_indent.pt, 0.0)
        if first_line_indent is not None and getattr(first_line_indent, "pt", None):
            indent_pt += max(first_line_indent.pt, 0.0)

    # 约 4pt 映射为 1 个空格，尽量保留层级感，但不把缩进放大得太夸张
    leading_spaces = max(0, int(round(indent_pt / 4.0)))
    return (" " * leading_spaces) + text


async def parse_uploaded_file(file: UploadFile) -> str:
    """
    解析上传的文件内容为字符串
    支持: .txt, .md, .csv, .pdf, .docx
    """
    filename = file.filename.lower()
    content = await file.read()
    file_text = ""

    try:
        if filename.endswith(('.txt', '.md', '.markdown', '.csv', '.json', '.py', '.js', '.html', '.css')):
            # 文本类文件直接解码
            file_text = content.decode('utf-8', errors='ignore')

        elif filename.endswith('.pdf'):
            try:
                pdf_file = io.BytesIO(content)
                reader = PdfReader(pdf_file)
                pages = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text.rstrip())
                file_text = "\n\n".join(pages)
            except Exception as e:
                return f"[系统提示: PDF 解析失败 - {str(e)}]"

        elif filename.endswith(('.doc', '.docx')):
            try:
                docx_file = io.BytesIO(content)
                doc = Document(docx_file)
                paragraphs = []
                for para in doc.paragraphs:
                    paragraphs.append(_docx_paragraph_to_text(para))
                file_text = "\n".join(paragraphs).rstrip("\n")
            except Exception as e:
                return f"[系统提示: Word 文档解析失败 - {str(e)}]"

        else:
            file_text = f"[系统提示: 不支持的文件格式 {file.filename}]"

    except Exception as e:
        file_text = f"[系统提示: 解析文件 {file.filename} 时发生错误 - {str(e)}]"

    return file_text.replace('\r\n', '\n').replace('\r', '\n')
