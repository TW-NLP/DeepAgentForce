/**
 * 知识库管理 JavaScript - 增强版
 */

// ============ 全局状态 ============
let allDocuments = [];
let selectedDocuments = new Set();
let searchKeyword = '';
let currentFilter = 'all';
let currentSort = 'recent';
let currentDrawerDocId = null;
let currentDrawerPreview = null;
let currentDrawerLoading = false;

// ============ UI 组件 ============
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
function getFileIcon(filename) {
    const ext = (filename || '').split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕', 'docx': '📘', 'doc': '📘',
        'txt': '📄', 'md': '📝', 'markdown': '📝', 'csv': '📊'
    };
    return icons[ext] || '📄';
}

function getFileTypeLabel(ext) {
    const labels = {
        'pdf': 'PDF文档', 'docx': 'Word文档', 'doc': 'Word文档',
        'txt': '文本文件', 'md': 'Markdown', 'csv': 'CSV表格'
    };
    return labels[ext] || '未知格式';
}

function formatDate(dateString) {
    if (!dateString) return '未知时间';
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatRelativeTime(dateString) {
    if (!dateString) return '刚刚';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) return '未知时间';
    const diff = Date.now() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return `${minutes} 分钟前`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} 小时前`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days} 天前`;
    return formatDate(dateString);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getDocumentExtension(doc) {
    const ext = (doc?.metadata?.file_extension || doc?.name || '').split('.').pop().toLowerCase();
    return ext || 'unknown';
}

function getDocumentTypeLabel(doc) {
    const ext = getDocumentExtension(doc);
    return getFileTypeLabel(ext);
}

function getFilterLabel(filter) {
    const labels = {
        all: '全部',
        pdf: 'PDF',
        doc: 'Word',
        docx: 'Word',
        txt: '文本',
        md: 'Markdown',
        markdown: 'Markdown',
        csv: 'CSV'
    };
    return labels[filter] || filter.toUpperCase();
}

function buildDocumentSummary(doc) {
    const chunks = doc.chunks || 0;
    const sizeBytes = doc.metadata?.size_bytes || doc.size || 0;
    const pieces = [];
    pieces.push(`${chunks} 个文本块`);
    if (sizeBytes) pieces.push(formatFileSize(sizeBytes));
    if (doc.metadata?.author) pieces.push(doc.metadata.author);
    return pieces.join(' · ');
}

function getDrawerElements() {
    return {
        drawer: document.getElementById('kbDetailDrawer'),
        empty: document.getElementById('kbDrawerEmpty'),
        shell: document.getElementById('kbDrawerShell'),
        title: document.getElementById('kbDrawerTitle'),
        subtitle: document.getElementById('kbDrawerSubtitle'),
        icon: document.getElementById('kbDrawerIcon'),
        status: document.getElementById('kbDrawerStatus'),
        meta: document.getElementById('kbDrawerMeta'),
        preview: document.getElementById('kbDrawerPreview'),
        actions: document.getElementById('kbDrawerActions'),
        closeBtn: document.getElementById('kbDrawerCloseBtn')
    };
}

function renderDrawer(doc, previewData = null, loading = false) {
    const els = getDrawerElements();
    if (!els.drawer || !els.empty || !els.shell) return;

    if (!doc) {
        els.empty.style.display = 'flex';
        els.shell.style.display = 'none';
        currentDrawerDocId = null;
        currentDrawerPreview = null;
        els.drawer.classList.remove('has-selection');
        return;
    }

    els.empty.style.display = 'none';
    els.shell.style.display = 'flex';
    els.drawer.classList.add('has-selection');

    if (els.title) els.title.textContent = doc.name || '未命名文档';
    if (els.subtitle) {
        const extLabel = getDocumentTypeLabel(doc);
        els.subtitle.textContent = `${extLabel} · ${formatRelativeTime(doc.uploaded_at)} · ${buildDocumentSummary(doc)}`;
    }
    if (els.icon) els.icon.textContent = getFileIcon(doc.name);
    if (els.status) {
        els.status.textContent = loading ? '加载中' : (previewData ? `${previewData.chunks?.length || 0} 个块` : '已选中文档');
    }

    if (els.meta) {
        const metadata = doc.metadata || {};
        els.meta.innerHTML = `
            <div class="kb-meta-item">
                <span class="kb-meta-label">文档ID</span>
                <span class="kb-meta-value mono">${escapeHtml(doc.document_id)}</span>
            </div>
            <div class="kb-meta-item">
                <span class="kb-meta-label">大小</span>
                <span class="kb-meta-value">${escapeHtml(formatFileSize(metadata.size_bytes || doc.size || 0))}</span>
            </div>
            <div class="kb-meta-item">
                <span class="kb-meta-label">块数</span>
                <span class="kb-meta-value">${doc.chunks || 0}</span>
            </div>
            <div class="kb-meta-item">
                <span class="kb-meta-label">更新时间</span>
                <span class="kb-meta-value">${escapeHtml(formatDate(doc.uploaded_at))}</span>
            </div>
            ${metadata.author ? `
                <div class="kb-meta-item">
                    <span class="kb-meta-label">作者</span>
                    <span class="kb-meta-value">${escapeHtml(metadata.author)}</span>
                </div>
            ` : ''}
            ${metadata.category ? `
                <div class="kb-meta-item">
                    <span class="kb-meta-label">分类</span>
                    <span class="kb-meta-value">${escapeHtml(metadata.category)}</span>
                </div>
            ` : ''}
        `;
    }

    if (els.actions) {
        els.actions.innerHTML = `
            <button class="kb-drawer-btn" type="button" onclick="previewDocument('${doc.document_id}')">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                预览内容
            </button>
            <button class="kb-drawer-btn secondary" type="button" onclick="showDocumentDetail('${doc.document_id}')">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                详情
            </button>
            <button class="kb-drawer-btn danger" type="button" onclick="deleteDocument('${doc.document_id}')">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                删除
            </button>
        `;
    }

    if (els.preview) {
        if (loading) {
            els.preview.innerHTML = `
                <div class="kb-preview-loading">
                    <div class="kb-preview-spinner"></div>
                    <div>
                        <div class="kb-preview-loading-title">正在加载预览</div>
                        <div class="kb-preview-loading-text">请稍候，文档内容正在读取中。</div>
                    </div>
                </div>
            `;
        } else if (previewData && previewData.chunks) {
            els.preview.innerHTML = previewData.chunks.length ? previewData.chunks.map((chunk, i) => `
                <div class="kb-preview-chunk">
                    <div class="kb-preview-chunk-header">
                        <span>块 ${i + 1}</span>
                        <button class="kb-copy-btn" type="button" onclick="copyChunkContent(${i}, '${doc.document_id}')">复制</button>
                    </div>
                    <div class="kb-preview-chunk-body">${escapeHtml(chunk.content || chunk.text || '')}</div>
                </div>
            `).join('') : '<div class="kb-preview-empty">暂无可预览内容</div>';
        } else {
            els.preview.innerHTML = '<div class="kb-preview-empty">点击“预览内容”加载文本切片。</div>';
        }
    }
}

function openDocumentDrawer(docId, options = {}) {
    const doc = allDocuments.find(item => item.document_id === docId);
    if (!doc) {
        renderDrawer(null);
        return;
    }

    currentDrawerDocId = docId;
    currentDrawerPreview = null;
    const loading = options.loading !== false;
    renderDrawer(doc, null, loading);
    document.querySelectorAll('.document-item').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-id') === docId);
    });
}

function closeDocumentDrawer() {
    currentDrawerDocId = null;
    currentDrawerPreview = null;
    renderDrawer(null);
    document.querySelectorAll('.document-item').forEach(item => item.classList.remove('active'));
}

function closeModal() {
    closeDocumentDrawer();
}

function getVisibleDocuments() {
    return allDocuments.filter(doc => {
        const name = (doc.name || '').toLowerCase();
        const keyword = searchKeyword.trim().toLowerCase();
        const ext = getDocumentExtension(doc);
        const matchesKeyword = !keyword || name.includes(keyword) || (doc.metadata?.title || '').toLowerCase().includes(keyword);
        const matchesFilter = currentFilter === 'all' || currentFilter === ext || (currentFilter === 'word' && ['doc', 'docx'].includes(ext));
        return matchesKeyword && matchesFilter;
    });
}

// ============ API 调用 ============
async function loadKnowledgeBaseStats() {
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/index/status`, { headers });
        if (!response.ok) {
            throw new Error(`请求失败 (${response.status})`);
        }
        const data = await response.json();

        if (data.success) {
            const statDocs = document.getElementById('statDocs');
            const statChunks = document.getElementById('statChunks');
            const statSize = document.getElementById('statSize');
            const statLastUpdate = document.getElementById('statLastUpdate');
            const kbHealthText = document.getElementById('kbHealthText');
            const kbHealthValue = document.getElementById('kbHealthValue');
            const kbTypeBreakdown = document.getElementById('kbTypeBreakdown');

            if (statDocs) statDocs.textContent = data.document_count;
            if (statChunks) statChunks.textContent = data.chunks_count || 0;
            if (statSize) statSize.textContent = data.total_size || '0 KB';
            if (statLastUpdate) statLastUpdate.textContent = data.last_updated ? formatDate(data.last_updated) : '暂无';
            if (kbHealthText) {
                kbHealthText.textContent = data.document_count > 0 ? '索引已就绪' : '等待上传文档';
            }
            if (kbHealthValue) {
                kbHealthValue.textContent = data.document_count > 0 ? 'Ready' : 'Empty';
            }
            if (kbTypeBreakdown) {
                const breakdown = data.file_type_counts || {};
                const entries = Object.entries(breakdown)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 4)
                    .map(([key, value]) => `<span class="kb-type-pill">${key} · ${value}</span>`)
                    .join('');
                kbTypeBreakdown.innerHTML = entries || '<span class="kb-type-pill muted">暂无分类</span>';
            }
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

async function loadDocuments() {
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents`, { headers });
        if (!response.ok) {
            throw new Error(`请求失败 (${response.status})`);
        }
        const data = await response.json();

        if (data.success) {
            allDocuments = (data.documents || []).sort((a, b) => {
                const at = new Date(a.uploaded_at || 0).getTime();
                const bt = new Date(b.uploaded_at || 0).getTime();
                return bt - at;
            });
            selectedDocuments = new Set(
                Array.from(selectedDocuments).filter(id => allDocuments.some(doc => doc.document_id === id))
            );
            if (currentDrawerDocId && !allDocuments.some(doc => doc.document_id === currentDrawerDocId)) {
                closeDocumentDrawer();
            }
            renderDocuments();
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
    }
}

function renderDocuments() {
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');
    const documentsCount = document.getElementById('documentsCount');
    const filteredCount = document.getElementById('filteredCount');
    const selectionBar = document.getElementById('selectionBar');
    const selectionText = document.getElementById('selectionText');

    let filteredDocs = getVisibleDocuments();

    if (currentSort === 'oldest') {
        filteredDocs = filteredDocs.sort((a, b) => new Date(a.uploaded_at || 0) - new Date(b.uploaded_at || 0));
    } else {
        filteredDocs = filteredDocs.sort((a, b) => new Date(b.uploaded_at || 0) - new Date(a.uploaded_at || 0));
    }

    // 更新选中计数
    const selectedCount = selectedDocuments.size;
    const selectAllBtn = document.getElementById('selectAllBtn');
    const batchDeleteBtn = document.getElementById('batchDeleteBtn');
    if (selectAllBtn) {
        selectAllBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
            ${selectedCount > 0 ? `已选 ${selectedCount}` : '全选'}
        `;
    }
    if (batchDeleteBtn) {
        batchDeleteBtn.disabled = selectedCount === 0;
        batchDeleteBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            ${selectedCount > 0 ? `批量删除 (${selectedCount})` : '批量删除'}
        `;
    }
    if (selectionBar) {
        selectionBar.classList.toggle('visible', selectedCount > 0);
    }
    if (selectionText) {
        selectionText.textContent = selectedCount > 0 ? `已选中 ${selectedCount} 个文档` : '未选择文档';
    }
    if (documentsCount) {
        documentsCount.textContent = allDocuments.length;
    }
    if (filteredCount) {
        filteredCount.textContent = filteredDocs.length;
    }

    if (filteredDocs.length === 0) {
        documentsList.innerHTML = '';
        if (emptyState) emptyState.style.display = 'flex';
        return;
    }

    if (emptyState) emptyState.style.display = 'none';

    documentsList.innerHTML = filteredDocs.map((doc, index) => `
        <div class="document-item ${selectedDocuments.has(doc.document_id) ? 'selected' : ''} ${currentDrawerDocId === doc.document_id ? 'active' : ''}"
             data-id="${doc.document_id}"
             style="animation-delay: ${index * 50}ms">
            <div class="doc-top">
                <div class="doc-checkbox" onclick="toggleDocumentSelect('${doc.document_id}', event)" aria-label="选择文档">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                </div>
                <div class="doc-icon">${getFileIcon(doc.name)}</div>
                <div class="doc-info" onclick="showDocumentDetail('${doc.document_id}')">
                    <div class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
                    <div class="doc-summary">${escapeHtml(buildDocumentSummary(doc))}</div>
                    <div class="doc-meta">
                        <span class="doc-type-badge">${getDocumentTypeLabel(doc)}</span>
                        ${doc.metadata?.category ? `<span class="doc-type-badge muted">${escapeHtml(doc.metadata.category)}</span>` : ''}
                    </div>
                </div>
                <div class="doc-actions">
                    <button class="doc-action-btn" onclick="previewDocument('${doc.document_id}')" title="预览">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                    </button>
                    <button class="doc-action-btn" onclick="showDocumentDetail('${doc.document_id}')" title="详情">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    </button>
                    <button class="doc-action-btn delete" onclick="deleteDocument('${doc.document_id}')" title="删除">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                </div>
            </div>
            <div class="doc-chips">
                <div class="doc-chip">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                    ${doc.chunks || 0} 块
                </div>
                <div class="doc-chip">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    ${formatRelativeTime(doc.uploaded_at)}
                </div>
                <div class="doc-chip">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 2h10l4 4v16H7z"/><path d="M14 2v6h6"/></svg>
                    ${formatFileSize(doc.metadata?.size_bytes || doc.size || 0)}
                </div>
            </div>
        </div>
    `).join('');
}

function toggleDocumentSelect(docId, event) {
    event.stopPropagation();
    if (selectedDocuments.has(docId)) {
        selectedDocuments.delete(docId);
    } else {
        selectedDocuments.add(docId);
    }
    renderDocuments();
}

function selectAllDocuments() {
    const visibleDocs = getVisibleDocuments();
    const visibleIds = new Set(visibleDocs.map(doc => doc.document_id));
    const allVisibleSelected = visibleDocs.length > 0 && visibleDocs.every(doc => selectedDocuments.has(doc.document_id));

    if (allVisibleSelected) {
        visibleIds.forEach(id => selectedDocuments.delete(id));
    } else {
        visibleDocs.forEach(doc => selectedDocuments.add(doc.document_id));
    }

    renderDocuments();
}

function batchDeleteDocuments() {
    if (selectedDocuments.size === 0) {
        showToast('请先选择要删除的文档', 'error');
        return;
    }

    if (!confirm(`确定要删除选中的 ${selectedDocuments.size} 个文档吗？`)) return;

    LoadingManager.show('正在批量删除...');

    const deletePromises = Array.from(selectedDocuments).map(docId =>
        fetch(`${getApiUrl()}/rag/documents/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        }).then(res => res.json())
    );

    Promise.all(deletePromises).then(results => {
        const successCount = results.filter(r => r.success).length;
        selectedDocuments.clear();
        loadDocuments();
        loadKnowledgeBaseStats();
        showToast(`成功删除 ${successCount} 个文档`);
    }).catch(err => {
        showToast('删除失败: ' + err.message, 'error');
    }).finally(() => {
        LoadingManager.hide();
    });
}

async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) return;

    LoadingManager.show('正在删除...');
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents/${documentId}`, {
            method: 'DELETE',
            headers
        });
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

function showDocumentDetail(docId) {
    openDocumentDrawer(docId, { loading: true });
    loadDocumentDrawerPreview(docId);
}

async function previewDocument(docId) {
    openDocumentDrawer(docId, { loading: true });
    await loadDocumentDrawerPreview(docId);
}

async function loadDocumentDrawerPreview(docId) {
    const doc = allDocuments.find(item => item.document_id === docId);
    if (!doc) return;
    currentDrawerLoading = true;
    renderDrawer(doc, null, true);
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`${getApiUrl()}/rag/documents/${docId}/content`, {
            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });

        if (!response.ok) throw new Error('无法加载预览');

        const data = await response.json();
        currentDrawerPreview = data;
        const latestDoc = allDocuments.find(item => item.document_id === docId);
        if (latestDoc && currentDrawerDocId === docId) {
            renderDrawer(latestDoc, data, false);
        }
    } catch (error) {
        const latestDoc = allDocuments.find(item => item.document_id === docId);
        if (latestDoc && currentDrawerDocId === docId) {
            renderDrawer(latestDoc, { chunks: [] }, false);
        }
        showToast('预览加载失败: ' + error.message, 'error');
    } finally {
        currentDrawerLoading = false;
    }
}

async function copyChunkContent(index, docId) {
    try {
        const doc = allDocuments.find(item => item.document_id === docId);
        if (!doc) return;
        const token = localStorage.getItem('access_token');
        const response = await fetch(`${getApiUrl()}/rag/documents/${docId}/content`, {
            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        const data = await response.json();
        const chunk = data?.chunks?.[index];
        if (!chunk) return;
        await navigator.clipboard.writeText(chunk.content || chunk.text || '');
        showToast('已复制文本块');
    } catch (error) {
        showToast('复制失败: ' + error.message, 'error');
    }
}

// ============ 上传进度管理 ============
const UploadProgress = {
    container: null,

    init() {
        if (this.container) return;

        const div = document.createElement('div');
        div.className = 'upload-progress-container';
        div.innerHTML = `
            <div class="upload-progress-header">
                <span class="upload-title">📤 正在上传文件...</span>
                <span class="upload-count">0/0</span>
            </div>
            <div class="upload-progress-bar">
                <div class="upload-progress-fill"></div>
            </div>
            <div class="upload-file-list"></div>
        `;
        document.body.appendChild(div);
        this.container = div;
        this.fileList = div.querySelector('.upload-file-list');
        this.fill = div.querySelector('.upload-progress-fill');
        this.count = div.querySelector('.upload-count');
    },

    show(total) {
        this.init();
        this.total = total;
        this.completed = 0;
        this.success = 0;
        this.failed = 0;
        this.fileList.innerHTML = '';
        this.container.classList.add('active');
    },

    update(current, filename, success) {
        this.completed = current;
        this.count.textContent = `${current}/${this.total}`;

        const percent = (current / this.total) * 100;
        this.fill.style.width = `${percent}%`;

        // 添加文件项
        const fileItem = document.createElement('div');
        fileItem.className = `upload-file-item ${success ? 'success' : 'failed'}`;
        fileItem.innerHTML = `
            <span class="file-status">${success ? '✅' : '❌'}</span>
            <span class="file-name">${escapeHtml(filename)}</span>
        `;
        this.fileList.appendChild(fileItem);

        if (success) this.success++;
        else this.failed++;

        // 保持滚动到底部
        this.fileList.scrollTop = this.fileList.scrollHeight;
    },

    hide() {
        if (this.container) {
            setTimeout(() => {
                this.container.classList.remove('active');
                if (this.failed === 0) {
                    showToast(`✅ 上传成功！${this.success} 个文件`);
                } else if (this.success === 0) {
                    showToast(`❌ 上传失败，请检查文件格式`, 'error');
                } else {
                    showToast(`⚠️ ${this.success} 成功，${this.failed} 失败`, 'warning');
                }
            }, 500);
        }
    }
};

// ============ 文件上传 ============
async function uploadSingleFile(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents/upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        const data = await response.json();
        if (!data.success) throw new Error(data.message || '上传失败');
        if (onProgress) onProgress(true);
        return true;
    } catch (error) {
        console.error(`上传 ${file.name} 失败:`, error);
        if (onProgress) onProgress(false);
        return false;
    }
}

async function handleBatchUpload(files) {
    if (files.length === 0) return;

    UploadProgress.show(files.length);

    const total = files.length;

    for (let i = 0; i < total; i++) {
        await uploadSingleFile(files[i], (success) => {
            UploadProgress.update(i + 1, files[i].name, success);
        });
    }

    UploadProgress.hide();

    // 刷新列表
    await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);
}

async function refreshKnowledgeBase() {
    LoadingManager.show('正在刷新知识库...');
    try {
        await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);
        showToast('知识库已刷新');
    } finally {
        LoadingManager.hide();
    }
}

// ============ 事件监听 ============
document.addEventListener('DOMContentLoaded', () => {
    LoadingManager.init();
    UploadProgress.init();

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('documentInput');
    const searchInput = document.getElementById('docSearchInput');
    const refreshBtn = document.getElementById('refreshKnowledgeBtn');
    const sortSelect = document.getElementById('docSortSelect');

    // 文件上传
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

            const invalidFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return !supportedExtensions.includes(ext);
            });

            if (invalidFiles.length > 0) {
                showToast(`已过滤 ${invalidFiles.length} 个不支持的文件`, 'warning');
            }

            if (validFiles.length > 0) {
                handleBatchUpload(validFiles);
            }
        });
    }

    // 搜索
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchKeyword = e.target.value;
            renderDocuments();
        });
    }

    // 排序
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            currentSort = e.target.value;
            renderDocuments();
        });
    }

    // 刷新
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshKnowledgeBase);
    }

    // 类型筛选
    document.querySelectorAll('[data-kb-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
            currentFilter = btn.getAttribute('data-kb-filter') || 'all';
            document.querySelectorAll('[data-kb-filter]').forEach(item => item.classList.remove('active'));
            btn.classList.add('active');
            renderDocuments();
        });
    });

    // 批量操作按钮
    const selectAllBtn = document.getElementById('selectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllDocuments);
    }

    const batchDeleteBtn = document.getElementById('batchDeleteBtn');
    if (batchDeleteBtn) {
        batchDeleteBtn.addEventListener('click', batchDeleteDocuments);
    }

    console.log('✅ knowledge.js 增强版已加载');
});

// 暴露给全局
window.deleteDocument = deleteDocument;
window.toggleDocumentSelect = toggleDocumentSelect;
window.showDocumentDetail = showDocumentDetail;
window.previewDocument = previewDocument;
window.closeModal = closeModal;
window.closeDocumentDrawer = closeDocumentDrawer;
