/**
 * Output 文件浏览器模块
 */

// ============ 状态变量 ============
let currentOutputPath = '';  // 当前目录路径
let currentPreviewFile = null;  // 当前预览的文件
let isOutputPanelOpen = false;

// ============ DOM 元素 ============
let outputToggleBtn, outputPanel, outputOverlay, outputFileList, outputBreadcrumb;
let outputPreviewModal, outputPreviewContent, outputPreviewFileName, outputDownloadBtn;
let outputPreviewClose, outputPreviewCloseBtn;

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initEventListeners();
    console.log('✅ output.js 已加载');
});

function initElements() {
    outputToggleBtn = document.getElementById('outputToggleBtn');
    outputPanel = document.getElementById('outputPanel');
    outputOverlay = document.getElementById('outputOverlay');
    outputFileList = document.getElementById('outputFileList');
    outputBreadcrumb = document.getElementById('outputBreadcrumb');
    outputPreviewModal = document.getElementById('outputPreviewModal');
    outputPreviewContent = document.getElementById('outputPreviewContent');
    outputPreviewFileName = document.getElementById('outputPreviewFileName');
    outputDownloadBtn = document.getElementById('outputDownloadBtn');
    outputPreviewClose = document.getElementById('outputPreviewClose');
    outputPreviewCloseBtn = document.getElementById('outputPreviewCloseBtn');
}

function initEventListeners() {
    // 切换按钮
    if (outputToggleBtn) {
        outputToggleBtn.addEventListener('click', toggleOutputPanel);
    }

    // 关闭按钮
    if (document.getElementById('outputPanelClose')) {
        document.getElementById('outputPanelClose').addEventListener('click', closeOutputPanel);
    }

    // 遮罩层点击
    if (outputOverlay) {
        outputOverlay.addEventListener('click', closeOutputPanel);
    }

    // 预览模态框关闭
    if (outputPreviewClose) {
        outputPreviewClose.addEventListener('click', closePreviewModal);
    }
    if (outputPreviewCloseBtn) {
        outputPreviewCloseBtn.addEventListener('click', closePreviewModal);
    }
    if (outputPreviewModal) {
        outputPreviewModal.addEventListener('click', (e) => {
            if (e.target === outputPreviewModal) {
                closePreviewModal();
            }
        });
    }

    // 下载按钮
    if (outputDownloadBtn) {
        outputDownloadBtn.addEventListener('click', downloadCurrentFile);
    }

    // 面包屑点击事件（使用事件委托）
    if (outputBreadcrumb) {
        outputBreadcrumb.addEventListener('click', (e) => {
            const item = e.target.closest('.output-breadcrumb-item');
            if (item) {
                const path = item.dataset.path || '';
                navigateToPath(path);
            }
        });
    }
}

// ============ 面板控制 ============
function toggleOutputPanel() {
    if (isOutputPanelOpen) {
        closeOutputPanel();
    } else {
        openOutputPanel();
    }
}

function openOutputPanel() {
    isOutputPanelOpen = true;
    if (outputPanel) {
        outputPanel.classList.add('open');
    }
    if (outputOverlay) {
        outputOverlay.classList.add('show');
    }
    if (outputToggleBtn) {
        outputToggleBtn.classList.add('active');
    }
    // 加载文件列表
    loadOutputFiles('');
}

function closeOutputPanel() {
    isOutputPanelOpen = false;
    if (outputPanel) {
        outputPanel.classList.remove('open');
    }
    if (outputOverlay) {
        outputOverlay.classList.remove('show');
    }
    if (outputToggleBtn) {
        outputToggleBtn.classList.remove('active');
    }
}

// ============ 文件列表加载 ============
async function loadOutputFiles(path = '') {
    if (!outputFileList) return;

    currentOutputPath = path;

    // 显示加载状态
    outputFileList.innerHTML = `
        <div class="output-loading">
            <div class="output-loading-spinner"></div>
            <span>加载中...</span>
        </div>
    `;

    try {
        const apiUrl = window.getApiUrl ? window.getApiUrl() : `${window.location.protocol}//${window.location.hostname}:8000/api`;
        const url = path ? `${apiUrl}/output/files?path=${encodeURIComponent(path)}` : `${apiUrl}/output/files`;

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        renderFileList(data.files);
        renderBreadcrumb(path);

    } catch (error) {
        console.error('加载文件列表失败:', error);
        outputFileList.innerHTML = `
            <div class="output-empty">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                <div class="output-empty-text">加载失败: ${error.message}</div>
            </div>
        `;
    }
}

function renderFileList(files) {
    if (!outputFileList) return;

    if (!files || files.length === 0) {
        outputFileList.innerHTML = `
            <div class="output-empty">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                <div class="output-empty-text">此目录为空</div>
            </div>
        `;
        return;
    }

    outputFileList.innerHTML = files.map(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        const iconClass = file.is_directory ? 'folder' : getIconClass(ext);
        const icon = file.is_directory
            ? getFolderIcon()
            : getFileIcon(ext);

        const size = file.is_directory ? '' : formatFileSize(file.size);
        const time = formatTime(file.modified_at);

        return `
            <div class="output-file-item" data-path="${file.path}" data-is-dir="${file.is_directory}" data-name="${file.name}">
                <div class="output-file-icon ${iconClass}">
                    ${icon}
                </div>
                <div class="output-file-info">
                    <div class="output-file-name" title="${file.name}">${file.name}</div>
                    <div class="output-file-meta">${size}${size && time ? ' · ' : ''}${time}</div>
                </div>
                <div class="output-file-actions">
                    ${!file.is_directory ? `
                        <button class="output-action-btn" onclick="event.stopPropagation(); previewFile('${file.path}', '${file.name}')" title="预览">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        </button>
                        <button class="output-action-btn" onclick="event.stopPropagation(); downloadFile('${file.path}')" title="下载">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');

    // 添加点击事件
    outputFileList.querySelectorAll('.output-file-item').forEach(item => {
        item.addEventListener('click', () => {
            const isDir = item.dataset.isDir === 'true';
            const path = item.dataset.path;

            if (isDir) {
                navigateToPath(path);
            } else {
                const name = item.dataset.name;
                previewFile(path, name);
            }
        });
    });
}

function renderBreadcrumb(path) {
    if (!outputBreadcrumb) return;

    const parts = path ? path.split('/').filter(Boolean) : [];
    const crumbs = [
        { name: 'output', path: '' }
    ];

    let currentPath = '';
    parts.forEach(part => {
        currentPath += (currentPath ? '/' : '') + part;
        crumbs.push({ name: part, path: currentPath });
    });

    outputBreadcrumb.innerHTML = crumbs.map((crumb, index) => {
        const isLast = index === crumbs.length - 1;
        return `
            ${index > 0 ? '<span class="output-breadcrumb-separator">/</span>' : ''}
            <span class="output-breadcrumb-item ${isLast ? 'output-breadcrumb-current' : ''}" data-path="${crumb.path}">
                ${crumb.name}
            </span>
        `;
    }).join('');
}

// ============ 导航 ============
function navigateToPath(path) {
    loadOutputFiles(path);
}

// ============ 文件预览 ============
async function previewFile(path, name) {
    if (!outputPreviewModal) return;

    currentPreviewFile = { path, name };

    // 显示模态框
    outputPreviewModal.classList.add('show');
    outputPreviewFileName.textContent = name;
    outputPreviewContent.textContent = '正在加载...';

    try {
        const apiUrl = window.getApiUrl ? window.getApiUrl() : `${window.location.protocol}//${window.location.hostname}:8000/api`;
        const response = await fetch(`${apiUrl}/output/files/preview?path=${encodeURIComponent(path)}`);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const content = await response.text();
        outputPreviewContent.textContent = content;

    } catch (error) {
        console.error('预览文件失败:', error);
        outputPreviewContent.textContent = `预览失败: ${error.message}`;
    }
}

function closePreviewModal() {
    if (outputPreviewModal) {
        outputPreviewModal.classList.remove('show');
    }
    currentPreviewFile = null;
}

// ============ 文件下载 ============
function downloadFile(path) {
    const apiUrl = window.getApiUrl ? window.getApiUrl() : `${window.location.protocol}//${window.location.hostname}:8000/api`;
    const url = `${apiUrl}/output/files/download?path=${encodeURIComponent(path)}`;

    // 创建一个隐藏的 <a> 标签进行下载
    const a = document.createElement('a');
    a.href = url;
    a.download = '';  // 使用服务器返回的文件名
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function downloadCurrentFile() {
    if (currentPreviewFile) {
        downloadFile(currentPreviewFile.path);
    }
}

// ============ 辅助函数 ============
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatTime(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        const now = new Date();
        const diff = now - date;

        // 如果是今天
        if (diff < 24 * 60 * 60 * 1000 && date.getDate() === now.getDate()) {
            return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        }

        // 如果是今年
        if (date.getFullYear() === now.getFullYear()) {
            return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
        }

        // 其他情况显示年份
        return date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'short', day: 'numeric' });
    } catch {
        return '';
    }
}

function getIconClass(ext) {
    const iconMap = {
        'txt': 'txt',
        'md': 'txt',
        'markdown': 'txt',
        'py': 'file',
        'js': 'file',
        'ts': 'file',
        'json': 'file',
        'csv': 'file',
        'pdf': 'file',
        'doc': 'file',
        'docx': 'file',
        'xls': 'file',
        'xlsx': 'file',
        'png': 'file',
        'jpg': 'file',
        'jpeg': 'file',
        'gif': 'file',
        'zip': 'file',
        'rar': 'file',
    };
    return iconMap[ext] || 'file';
}

function getFileIcon(ext) {
    const iconMap = {
        'txt': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
        'md': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>',
        'py': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
        'json': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
        'csv': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/></svg>',
    };
    return iconMap[ext] || '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
}

function getFolderIcon() {
    return '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';
}

// 将函数暴露到全局作用域，供 HTML 内联 onclick 调用
window.previewFile = previewFile;
window.downloadFile = downloadFile;
