from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from typing import List, Optional
import io
from pypdf import PdfReader
from docx import Document

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
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        file_text += text + "\n"
            except Exception as e:
                return f"[系统提示: PDF 解析失败 - {str(e)}]"

        elif filename.endswith(('.doc', '.docx')):
            try:
                docx_file = io.BytesIO(content)
                doc = Document(docx_file)
                for para in doc.paragraphs:
                    if para.text.strip():
                        file_text += para.text + "\n"
            except Exception as e:
                return f"[系统提示: Word 文档解析失败 - {str(e)}]"

        else:
            file_text = f"[系统提示: 不支持的文件格式 {file.filename}]"

    except Exception as e:
        file_text = f"[系统提示: 解析文件 {file.filename} 时发生错误 - {str(e)}]"

    return file_text


