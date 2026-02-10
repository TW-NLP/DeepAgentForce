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
            if PdfReader is None:
                return f"[系统提示: 未安装 pypdf 库，无法解析 {file.filename}]"
            # 处理 PDF
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    file_text += text + "\n"
        
        elif filename.endswith(('.doc', '.docx')):
            if Document is None:
                return f"[系统提示: 未安装 python-docx 库，无法解析 {file.filename}]"
            # 处理 Word
            docx_file = io.BytesIO(content)
            doc = Document(docx_file)
            for para in doc.paragraphs:
                file_text += para.text + "\n"
        
        else:
            file_text = f"[系统提示: 不支持的文件格式 {file.filename}]"

    except Exception as e:
        file_text = f"[系统提示: 解析文件 {file.filename} 时发生错误]"
    
    # 包装一下文件内容，让 AI 知道这是文件
    return f"\n\n=== 📎 附件文件内容: {file.filename} ===\n{file_text}\n=== 附件结束 ===\n\n"


