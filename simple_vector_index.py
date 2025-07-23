#!/usr/bin/env python3
"""
简化版向量索引系统 - 避免复杂依赖
"""

import os
import json
import hashlib
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

# 基础库
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 文档处理
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class SimpleVectorIndex:
    def __init__(self, downloads_dir: str = "./documents", 
                 index_dir: str = "./.vector_index"):
        """简化版向量索引系统"""
        self.downloads_dir = Path(downloads_dir).expanduser()
        self.index_dir = Path(index_dir).expanduser()
        self.index_dir.mkdir(exist_ok=True)
        
        # 使用 TF-IDF 向量化，针对中文优化
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # 不使用英文停用词，支持中文
            ngram_range=(1, 3),  # 扩展到3-gram，更好支持中文词汇
            min_df=1,
            max_df=0.95,
            token_pattern=r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+',  # 支持中文字符和英文数字
            analyzer='char',  # 使用字符级分析，更适合中文
        )
        
        # 文档内容和向量
        self.documents = []
        self.document_vectors = None
        
        # 数据库
        self.db_path = self.index_dir / "metadata.db"
        self._init_database()
        
        print(f"简化版向量索引系统已初始化")
        print(f"文档目录: {self.downloads_dir}")
        print(f"索引目录: {self.index_dir}")
        
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                last_modified TIMESTAMP,
                content_hash TEXT,
                content_text TEXT,
                vector_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """从PDF提取文本"""
        if not PDF_AVAILABLE:
            return f"PDF文件: {file_path.name} (需要安装PyPDF2)"
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            return f"PDF文件: {file_path.name} (读取失败: {str(e)})"
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """从Word文档提取文本"""
        if not DOCX_AVAILABLE:
            return f"Word文档: {file_path.name} (需要安装python-docx)"
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            return f"Word文档: {file_path.name} (读取失败: {str(e)})"
    
    def _extract_text_from_image(self, file_path: Path) -> str:
        """从图片提取文本"""
        if not OCR_AVAILABLE:
            return f"图片文件: {file_path.name} (需要安装PIL和pytesseract)"
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return f"图片文件: {file_path.name}\n提取文本: {text}"
        except Exception as e:
            return f"图片文件: {file_path.name} (OCR失败: {str(e)})"
    
    def _extract_text_from_csv(self, file_path: Path) -> str:
        """从CSV提取信息"""
        if not PANDAS_AVAILABLE:
            return f"CSV文件: {file_path.name} (需要安装pandas)"
        
        try:
            df = pd.read_csv(file_path, nrows=50)
            info = f"CSV文件: {file_path.name}\n"
            info += f"列名: {', '.join(df.columns.tolist())}\n"
            info += f"行数: {len(df)}\n"
            info += f"数据预览:\n{df.head().to_string()}"
            return info
        except Exception as e:
            return f"CSV文件: {file_path.name} (读取失败: {str(e)})"
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """从文件提取文本内容"""
        file_ext = file_path.suffix.lower()
        
        # 根据文件类型处理
        if file_ext == '.pdf':
            content = self._extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            content = self._extract_text_from_docx(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            content = self._extract_text_from_image(file_path)
        elif file_ext == '.csv':
            content = self._extract_text_from_csv(file_path)
        elif file_ext in ['.txt', '.md', '.py', '.js', '.json', '.xml', '.html']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                content = f"文本文件: {file_path.name} (读取失败: {str(e)})"
        else:
            # 默认处理
            content = f"文件: {file_path.name}\n类型: {file_ext}\n大小: {file_path.stat().st_size} bytes"
        
        # 添加文件路径信息
        full_content = f"文件路径: {file_path}\n文件名: {file_path.name}\n{content}"
        return full_content
    
    def build_index(self, max_files: int = 500):
        """构建索引"""
        print("开始构建索引...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清空旧数据
        cursor.execute("DELETE FROM file_metadata")
        self.documents = []
        
        file_count = 0
        processed_count = 0
        
        # 遍历文件
        for file_path in self.downloads_dir.rglob('*'):
            if file_path.is_file() and processed_count < max_files:
                file_count += 1
                
                # 跳过隐藏文件和索引目录
                if file_path.name.startswith('.') or '.vector_index' in str(file_path):
                    continue
                
                try:
                    # 提取文本内容
                    content = self._extract_text_from_file(file_path)
                    if not content.strip():
                        continue
                    
                    # 添加到文档列表
                    self.documents.append(content)
                    vector_index = len(self.documents) - 1
                    
                    # 保存元数据
                    file_stat = file_path.stat()
                    cursor.execute("""
                        INSERT INTO file_metadata 
                        (file_path, file_name, file_size, file_type, last_modified, 
                         content_text, vector_index)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(file_path),
                        file_path.name,
                        file_stat.st_size,
                        file_path.suffix.lower(),
                        datetime.fromtimestamp(file_stat.st_mtime),
                        content[:2000],  # 只存储前2000字符
                        vector_index
                    ))
                    
                    processed_count += 1
                    if processed_count % 50 == 0:
                        print(f"已处理 {processed_count} 个文件...")
                        
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {e}")
                    continue
        
        if self.documents:
            # 构建 TF-IDF 向量
            print("正在构建TF-IDF向量...")
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            
            # 保存向量化器和文档
            data_to_save = {
                'vectorizer': self.vectorizer,
                'documents': self.documents,
                'document_vectors': self.document_vectors
            }
            
            with open(self.index_dir / "vectors.pkl", 'wb') as f:
                pickle.dump(data_to_save, f)
        
        conn.commit()
        conn.close()
        
        print(f"索引构建完成！")
        print(f"总文件数: {file_count}")
        print(f"成功处理: {processed_count}")
        print(f"向量维度: {self.document_vectors.shape if self.document_vectors is not None else 0}")
    
    def load_index(self):
        """加载索引"""
        vectors_path = self.index_dir / "vectors.pkl"
        if vectors_path.exists():
            with open(vectors_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.documents = data['documents']
                self.document_vectors = data['document_vectors']
            print(f"已加载索引，包含 {len(self.documents)} 个文档")
        else:
            print("未找到已保存的索引，请先运行 build_index()")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索 - 结合向量相似度和关键词匹配"""
        if not self.documents:
            print("索引为空，请先构建索引")
            return []
        
        # 向量化查询
        query_vector = self.vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # 获取详细信息
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 同时进行数据库文本搜索作为补充
        cursor.execute("""
            SELECT file_path, file_name, file_type, file_size, 
                   last_modified, content_text, vector_index
            FROM file_metadata 
            WHERE content_text LIKE ?
        """, (f'%{query}%',))
        
        text_matches = cursor.fetchall()
        
        # 合并结果
        results = []
        processed_indices = set()
        
        # 先添加向量搜索结果
        top_indices = np.argsort(similarities)[::-1][:top_k*2]  # 扩展搜索范围
        
        for idx in top_indices:
            score = similarities[idx]
            if score > 0:
                cursor.execute("""
                    SELECT file_path, file_name, file_type, file_size, 
                           last_modified, content_text 
                    FROM file_metadata 
                    WHERE vector_index = ?
                """, (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    results.append({
                        'score': float(score),
                        'file_path': row[0],
                        'file_name': row[1],
                        'file_type': row[2],
                        'file_size': row[3],
                        'last_modified': row[4],
                        'content_preview': row[5][:300] + "..." if len(row[5]) > 300 else row[5],
                        'match_type': 'vector'
                    })
                    processed_indices.add(idx)
        
        # 添加文本匹配结果（如果向量搜索没有找到）
        for row in text_matches:
            vector_idx = row[6]
            if vector_idx not in processed_indices:
                # 为文本匹配设置较高的基础分数
                text_score = 0.8  
                results.append({
                    'score': text_score,
                    'file_path': row[0],
                    'file_name': row[1],
                    'file_type': row[2],
                    'file_size': row[3],
                    'last_modified': row[4],
                    'content_preview': row[5][:300] + "..." if len(row[5]) > 300 else row[5],
                    'match_type': 'text'
                })
        
        # 按分数重新排序并返回top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        conn.close()
        return results[:top_k]
    
    def search_cli(self, query: str, top_k: int = 10):
        """命令行搜索"""
        print(f"\n🔍 搜索: '{query}'")
        print("=" * 60)
        
        results = self.search(query, top_k)
        
        if not results:
            print("❌ 未找到相关结果")
            return
        
        for i, result in enumerate(results, 1):
            match_type = result.get('match_type', 'vector')
            match_icon = "🎯" if match_type == 'text' else "📊"
            print(f"\n📄 结果 {i} {match_icon} (分数: {result['score']:.3f})")
            print(f"   文件: {result['file_name']}")
            print(f"   路径: {result['file_path']}")
            print(f"   类型: {result['file_type']}")
            print(f"   大小: {result['file_size']} bytes")
            print(f"   修改时间: {result['last_modified']}")
            print(f"   匹配类型: {'文本匹配' if match_type == 'text' else '向量相似度'}")
            print(f"   内容预览: {result['content_preview']}")
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="简化版文档向量索引")
    parser.add_argument("--build", action="store_true", help="构建索引")
    parser.add_argument("--search", type=str, help="搜索查询")
    parser.add_argument("--top_k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--max_files", type=int, default=500, help="最大处理文件数")
    parser.add_argument("--documents_dir", type=str, default="./documents", help="文档目录路径")
    
    args = parser.parse_args()
    
    # 初始化系统
    system = SimpleVectorIndex(downloads_dir=args.documents_dir)
    
    if args.build:
        system.build_index(args.max_files)
    elif args.search:
        system.load_index()
        system.search_cli(args.search, args.top_k)
    else:
        # 交互式模式
        system.load_index()
        print("\n🚀 简化版向量索引系统已启动！")
        print("输入查询内容进行搜索，输入 'quit' 退出")
        
        while True:
            query = input("\n🔍 请输入搜索查询: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                system.search_cli(query)

if __name__ == "__main__":
    main()