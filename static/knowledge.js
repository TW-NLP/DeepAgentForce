/**
 * 知识库管理 JavaScript
 */

// ============ UI 交互组件: Loading 管理 ============
const LoadingManager = {
    overlay: null,
    textElement: null,

    init() {
        if (this.overlay) return;

        const div = document.createElement('div');
        div.className = 'loading-overlay';
        div.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">正在处理...</div>
        `;
        document.body.appendChild(div);

        this.overlay = div;
        this.textElement = div.querySelector('.loading-text');
    },

    show(message = '正在加载...') {
        this.init();
        this.textElement.textContent = message;
        this.overlay.classList.add('active');
    },

    updateText(message) {
        if (this.textElement) {
            this.textElement.textContent = message;
        }
    },

    hide() {
        if (this.overlay) {
            this.overlay.classList.remove('active');
        }
    }
};

// ============ 工具函数 ============
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${type === 'success' ? '✅' : '❌'}</span>
        <span class="toast-message">${message}</span>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕', 'docx': '📘', 'doc': '📘',
        'txt': '📄', 'md': '📝', 'markdown': '📝', 'csv': '📊'
    };
    return icons[ext] || '📄';
}

function formatDate(dateString) {
    if (!dateString) return '未知时间';
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN');
}

// ============ API 调用 ============
async function loadKnowledgeBaseStats() {
    try {
        // 🆕 使用 authFetch
        const response = await authFetch(`${getApiUrl()}/rag/index/status`);
        const data = await response.json();

        if (data.success) {
            const docCountEl = document.getElementById('statDocs');
            if (docCountEl) docCountEl.textContent = data.document_count;
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

async function loadDocuments() {
    try {
        // 🆕 使用 authFetch
        const response = await authFetch(`${getApiUrl()}/rag/documents`);
        const data = await response.json();
        const documentsList = document.getElementById('documentsList');

        if (data.success && data.documents && data.documents.length > 0) {
            documentsList.innerHTML = data.documents.map(doc => `
                <div class="document-item">
                    <div class="doc-icon">${getFileIcon(doc.name)}</div>
                    <div class="doc-info">
                        <div class="doc-name" title="${doc.name}">${doc.name}</div>
                        <div class="doc-meta">
                            <span>📦 ${doc.chunks} 块</span>
                            <span>🕐 ${formatDate(doc.uploaded_at)}</span>
                        </div>
                    </div>
                    <div class="doc-actions">
                        <button class="doc-action-btn" onclick="deleteDocument('${doc.document_id}')" title="删除">🗑️</button>
                    </div>
                </div>
            `).join('');
        } else {
            documentsList.innerHTML = `<div class="empty-state">📭 暂无文档</div>`;
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
    }
}

async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) return;

    LoadingManager.show('正在删除...');
    try {
        // 🆕 使用 authFetch
        const response = await authFetch(`${getApiUrl()}/rag/documents/${documentId}`, { method: 'DELETE' });
        const data = await response.json();

        if (data.success) {
            showToast('文档已删除');
            await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);
        } else {
            throw new Error(data.message || '删除失败');
        }
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        LoadingManager.hide();
    }
}

// ============ 文件上传 ============
async function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 🆕 文件上传需要手动构造 Authorization header
        const token = window.auth && window.auth.getAccessToken ? window.auth.getAccessToken() : null;
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const response = await fetch(`${getApiUrl()}/rag/documents/upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.message);
        return true;
    } catch (error) {
        console.error(`上传 ${file.name} 失败:`, error);
        return false;
    }
}

async function handleBatchUpload(files) {
    if (files.length === 0) return;

    LoadingManager.show('准备上传...');

    let successCount = 0;
    const total = files.length;
    const failedFiles = [];

    for (let i = 0; i < total; i++) {
        const file = files[i];
        LoadingManager.updateText(`正在上传 (${i + 1}/${total}): ${file.name}`);

        const success = await uploadSingleFile(file);

        if (success) {
            successCount++;
        } else {
            failedFiles.push(file.name);
        }
    }

    LoadingManager.updateText('正在刷新列表...');
    await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);

    LoadingManager.hide();

    if (failedFiles.length === 0) {
        showToast(`✅ 全部成功！共上传 ${successCount} 个文件`);
    } else if (successCount === 0) {
        showToast(`❌ 全部失败，请检查网络或文件格式`, 'error');
    } else {
        showToast(`⚠️ 上传完成: ${successCount} 个成功，${failedFiles.length} 个失败`, 'warning');
    }
}

// ============ 事件监听 ============
document.addEventListener('DOMContentLoaded', () => {
    LoadingManager.init();

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            handleBatchUpload(files);
            fileInput.value = '';
        });
    }

    if (uploadArea) {
        ['dragover', 'dragleave'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                if (eventName === 'dragover') uploadArea.classList.add('dragging');
                else uploadArea.classList.remove('dragging');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');

            const files = Array.from(e.dataTransfer.files);
            const supportedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'];

            const validFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return supportedExtensions.includes(ext);
            });

            if (validFiles.length > 0) {
                handleBatchUpload(validFiles);
            }
        });
    }

    console.log('✅ knowledge.js 已加载');
});

// 暴露给全局
window.deleteDocument = deleteDocument;
