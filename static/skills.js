/**
 * Skill 管理 JavaScript
 * 处理 Skill 的增删改查、导入导出等功能
 */

// 使用全局函数动态获取 API 地址（避免函数名冲突）
function getSkillApiUrl() {
    if (window.getApiUrl) {
        return window.getApiUrl();
    }
    return `${window.location.protocol}//${window.location.hostname}:8000/api`;
}

// 🆕 获取带认证的请求头
function getSkillHeaders() {
    const token = localStorage.getItem('access_token');
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

// 🆕 带认证的 fetch 请求
async function authSkillFetch(url, options = {}) {
    const token = localStorage.getItem('access_token');
    const headers = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const mergedOptions = {
        ...options,
        headers: {
            ...headers,
            ...options.headers,
        },
    };

    let response = await fetch(url, mergedOptions);

    // 如果收到 401，尝试刷新 Token
    if (response.status === 401 && token) {
        const refreshed = await window.auth?.refreshAccessToken?.();
        if (refreshed) {
            headers['Authorization'] = `Bearer ${localStorage.getItem('access_token')}`;
            response = await fetch(url, { ...options, headers });
        } else {
            window.location.href = 'login.html';
            return null;
        }
    }

    return response;
}

// 全局变量
let skills = [];
let currentSkillId = null;
let scriptFields = [];
let isEditMode = false;

// ==================== 初始化 ====================

// 不要在这里自动加载，让 index.html 导航逻辑控制
// document.addEventListener('DOMContentLoaded', async () => {
//     console.log('🚀 Skill 页面加载中...');
//     await loadSkills();
//     console.log('✅ Skill 页面加载完成');
// });

async function loadSkills() {
    try {
        console.log('📡 正在请求 Skills API...');
        // 🆕 使用 authFetch 携带认证信息
        const response = await authSkillFetch(`${getSkillApiUrl()}/skills`);
        if (!response) return;  // 未登录

        console.log('📬 响应状态:', response.status);

        const data = await response.json();
        console.log('📦 获取到数据:', data);

        if (data.success) {
            skills = data.skills;
            renderSkills();
            updateStats();
            console.log('✅ Skills 渲染完成');
        } else {
            console.error('❌ API 返回失败:', data);
            showToast('加载 Skills 失败', 'error');
        }
    } catch (error) {
        console.error('❌ 请求失败:', error);
        showToast('无法连接到服务器: ' + error.message, 'error');
    }
}

function updateStats() {
    const total = skills.length;
    const totalScripts = skills.reduce((sum, s) => sum + (s.script_count || 0), 0);

    const statTotalEl = document.getElementById('statTotal');
    const statScriptsEl = document.getElementById('statScripts');

    if (statTotalEl) statTotalEl.textContent = total;
    if (statScriptsEl) statScriptsEl.textContent = totalScripts;
}

// ==================== 渲染 Skills ====================

function renderSkills() {
    const grid = document.getElementById('skillsGrid');
    grid.innerHTML = '';

    // 添加"添加 Skill"卡片
    const addCard = document.createElement('div');
    addCard.className = 'add-skill-card';
    addCard.onclick = () => openSkillModal();
    addCard.innerHTML = `
        <div class="add-skill-icon">➕</div>
        <div class="add-skill-title">添加新 Skill</div>
        <div class="add-skill-hint">上传自定义技能插件</div>
    `;
    grid.appendChild(addCard);

    // 渲染每个 Skill 卡片
    skills.forEach(skill => {
        const card = createSkillCard(skill);
        grid.appendChild(card);
    });
}

function createSkillCard(skill) {
    const card = document.createElement('div');
    card.className = 'skill-card';

    const isBuiltIn = ['pdf-processing', 'rag-query', 'web-search'].includes(skill.id);
    const icon = getSkillIcon(skill.id);

    card.innerHTML = `
        <div class="skill-card-header">
            <div class="skill-icon">${icon}</div>
            <div class="skill-info">
                <div class="skill-name">${skill.name}</div>
                <div class="skill-version">
                    <span>v${skill.version}</span>
                    ${isBuiltIn ? '<span class="badge">内置</span>' : ''}
                </div>
            </div>
        </div>
        <div class="skill-description">${skill.description || '暂无描述'}</div>
        <div class="skill-meta">
            <div class="skill-meta-item">
                <span>📜</span>
                <span>${skill.script_count} 个脚本</span>
            </div>
            <div class="skill-meta-item">
                <span>👤</span>
                <span>${skill.author || '未知'}</span>
            </div>
        </div>
        ${skill.tags && skill.tags.length > 0 ? `
            <div class="skill-tags">
                ${skill.tags.map(tag => `<span class="skill-tag">${tag}</span>`).join('')}
            </div>
        ` : ''}
        <div class="skill-card-actions">
            <button class="skill-action-btn" onclick="viewSkill('${skill.id}')">
                👁️ 查看
            </button>
            <button class="skill-action-btn primary" onclick="viewSkill('${skill.id}')">
                ✏️ 编辑
            </button>
            ${!isBuiltIn ? `
                <button class="skill-action-btn danger" onclick="confirmDelete('${skill.id}')">
                    🗑️
                </button>
            ` : ''}
        </div>
    `;

    return card;
}

function getSkillIcon(skillId) {
    const iconMap = {
        'pdf-processing': '📄',
        'rag-query': '🔍',
        'web-search': '🌐',
        'default': '🛠️'
    };
    return iconMap[skillId] || iconMap['default'];
}

// ==================== Modal 操作 ====================

function openSkillModal(skillId = null) {
    const modal = document.getElementById('skillModal');
    const title = document.getElementById('modalTitle');
    const form = document.getElementById('skillForm');

    // 重置表单
    form.reset();
    scriptFields = [];
    document.getElementById('scriptList').innerHTML = '';
    document.getElementById('validationResults').classList.remove('show');

    if (skillId) {
        // 编辑模式
        isEditMode = true;
        currentSkillId = skillId;
        title.textContent = '编辑 Skill';

        const skill = skills.find(s => s.id === skillId);
        if (skill) {
            loadSkillForEdit(skill);
        }
    } else {
        // 添加模式
        isEditMode = false;
        currentSkillId = null;
        title.textContent = '添加新 Skill';

        // 添加默认脚本字段
        addScriptField();
    }

    // 强制设置白色背景
    const modalInner = modal.querySelector('.modal');
    if (modalInner) {
        modalInner.style.backgroundColor = '#ffffff';
        const header = modalInner.querySelector('.modal-header');
        const body = modalInner.querySelector('.modal-body');
        const footer = modalInner.querySelector('.modal-footer');
        if (header) header.style.backgroundColor = '#ffffff';
        if (body) body.style.backgroundColor = '#ffffff';
        if (footer) footer.style.backgroundColor = '#ffffff';
    }

    modal.classList.add('active');
}

function closeSkillModal() {
    const modal = document.getElementById('skillModal');
    modal.classList.remove('active');
    isEditMode = false;
    currentSkillId = null;
}

async function loadSkillForEdit(skill) {
    document.getElementById('skillName').value = skill.name;
    document.getElementById('skillDescription').value = skill.description || '';
    document.getElementById('skillVersion').value = skill.version || '1.0.0';
    document.getElementById('skillAuthor').value = skill.author || '';
    document.getElementById('skillTags').value = (skill.tags || []).join(', ');

    // 加载 SKILL.md 内容
    try {
        // 🆕 使用 authFetch
        const response = await authSkillFetch(`${getSkillApiUrl()}/skills/${skill.id}/content`);
        if (!response) return;

        const data = await response.json();

        if (data.success) {
            document.getElementById('skillMdContent').value = data.skill_md || '';

            // 加载 scripts
            if (data.scripts) {
                Object.entries(data.scripts).forEach(([name, content]) => {
                    addScriptField(name, content);
                });
            }
        }
    } catch (error) {
        console.error('加载 Skill 内容失败:', error);
    }
}

// ==================== 脚本管理 ====================

function addScriptField(name = '', content = '') {
    const container = document.getElementById('scriptList');
    const index = scriptFields.length;

    scriptFields.push({ name, content });

    const div = document.createElement('div');
    div.className = 'script-item';
    div.dataset.index = index;
    div.innerHTML = `
        <div class="script-icon">🐍</div>
        <div class="script-info">
            <input type="text" class="form-input" placeholder="脚本文件名 (例如: main.py)"
                   value="${name}" data-field="name" style="margin-bottom: 8px;">
            <textarea class="form-textarea" placeholder="Python 脚本内容..."
                      data-field="content" style="min-height: 150px;">${content}</textarea>
        </div>
        <button type="button" class="script-add-btn" onclick="removeScriptField(${index})">✕</button>
    `;

    container.appendChild(div);
}

function removeScriptField(index) {
    const container = document.getElementById('scriptList');
    const items = container.querySelectorAll('.script-item');

    if (items[index]) {
        items[index].remove();
        scriptFields[index] = null;
    }
}

function collectScriptData() {
    const container = document.getElementById('scriptList');
    const items = container.querySelectorAll('.script-item');
    const scripts = {};

    items.forEach(item => {
        const nameInput = item.querySelector('[data-field="name"]');
        const contentInput = item.querySelector('[data-field="content"]');

        const name = nameInput?.value?.trim();
        const content = contentInput?.value;

        if (name && content) {
            scripts[name] = content;
        }
    });

    return scripts;
}

// ==================== Skill 验证与安装 ====================

async function validateSkill() {
    const skillName = document.getElementById('skillName').value.trim();
    const skillMdContent = document.getElementById('skillMdContent').value;
    const scripts = collectScriptData();
    const validationResults = document.getElementById('validationResults');

    try {
        const formData = new FormData();
        formData.append('skill_md', skillMdContent);
        formData.append('scripts', JSON.stringify(scripts));

        const response = await fetch(`${getSkillApiUrl()}/skills/validate`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // 显示验证结果
        validationResults.classList.add('show');

        if (result.valid) {
            validationResults.className = 'validation-results show success';
            validationResults.innerHTML = `
                <div class="validation-title">✅ 验证通过</div>
                <ul class="validation-list">
                    ${result.warnings.map(w => `<li>⚠️ ${w}</li>`).join('')}
                    ${!result.warnings.length ? '<li>无警告</li>' : ''}
                </ul>
            `;
        } else {
            validationResults.className = 'validation-results show error';
            validationResults.innerHTML = `
                <div class="validation-title">❌ 验证失败</div>
                <ul class="validation-list">
                    ${result.errors.map(e => `<li>${e}</li>`).join('')}
                </ul>
            `;
        }
    } catch (error) {
        console.error('验证失败:', error);
        showToast('验证请求失败', 'error');
    }
}

async function installSkill() {
    const skillName = document.getElementById('skillName').value.trim();
    const skillDescription = document.getElementById('skillDescription').value.trim();
    const skillVersion = document.getElementById('skillVersion').value.trim() || '1.0.0';
    const skillAuthor = document.getElementById('skillAuthor').value.trim();
    const skillTags = document.getElementById('skillTags').value.trim();
    const skillMdContent = document.getElementById('skillMdContent').value;
    const scripts = collectScriptData();

    if (!skillName) {
        showToast('请输入 Skill 名称', 'warning');
        return;
    }

    if (!skillMdContent) {
        showToast('请输入 SKILL.md 内容', 'warning');
        return;
    }

    // 构建 SKILL.md (确保包含必要字段)
    let finalSkillMd = skillMdContent;

    // 如果用户没有在 SKILL.md 中设置 name/description，注入进去
    if (!finalSkillMd.includes('name:')) {
        finalSkillMd = `---\nname: ${skillName}\ndescription: ${skillDescription}\nversion: ${skillVersion}\nauthor: ${skillAuthor}\ntags:\n  - ${(skillTags.split(',')[0] || 'custom').trim()}\n---\n\n${finalSkillMd}`;
    }

    try {
        const formData = new FormData();
        formData.append('skill_name', skillName);
        formData.append('skill_md', finalSkillMd);
        formData.append('scripts', JSON.stringify(scripts));
        formData.append('force', isEditMode ? 'true' : 'false');

        // 🆕 添加 Authorization header
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const response = await fetch(`${getSkillApiUrl()}/skills/install`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');
            closeSkillModal();
            await loadSkills();
        } else {
            showToast(result.message || '安装失败', 'error');
            if (result.errors) {
                console.error('安装错误:', result.errors);
            }
        }
    } catch (error) {
        console.error('安装失败:', error);
        showToast('安装请求失败', 'error');
    }
}

// ==================== Skill 查看/编辑/删除 ====================

async function viewSkill(skillId) {
    currentSkillId = skillId;

    try {
        // 🆕 使用 authFetch
        const response = await authSkillFetch(`${getSkillApiUrl()}/skills/${skillId}/content`);
        if (!response) return;

        const data = await response.json();

        if (data.success) {
            openViewModal(skillId, data);
        } else {
            showToast('加载 Skill 详情失败', 'error');
        }
    } catch (error) {
        console.error('加载 Skill 详情失败:', error);
        showToast('无法加载 Skill 详情', 'error');
    }
}

function openViewModal(skillId, data) {
    const modal = document.getElementById('viewModal');
    const title = document.getElementById('viewModalTitle');
    const body = document.getElementById('viewModalBody');

    const skill = skills.find(s => s.id === skillId);
    if (!skill) return;

    title.textContent = skill.name;

    // 渲染内容
    let scriptsHtml = '';
    if (data.scripts && Object.keys(data.scripts).length > 0) {
        scriptsHtml = `
            <div class="form-section">
                <div class="form-section-title">📜 脚本列表</div>
                <div class="script-list">
                    ${Object.entries(data.scripts).map(([name, content]) => `
                        <div class="script-item" style="flex-direction: column; align-items: stretch;">
                            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                                <div class="script-icon">🐍</div>
                                <div class="script-name">${name}</div>
                            </div>
                            <textarea class="form-textarea" readonly style="min-height: 200px; background: #1e1e1e; color: #d4d4d4;">${escapeHtml(content)}</textarea>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    body.innerHTML = `
        <div class="form-section">
            <div class="form-section-title">📝 基本信息</div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">名称</label>
                    <div style="padding: 8px 0;">${skill.name}</div>
                </div>
                <div class="form-group">
                    <label class="form-label">版本</label>
                    <div style="padding: 8px 0;">v${skill.version}</div>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">描述</label>
                <div style="padding: 8px 0;">${skill.description || '暂无描述'}</div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">作者</label>
                    <div style="padding: 8px 0;">${skill.author || '未知'}</div>
                </div>
                <div class="form-group">
                    <label class="form-label">脚本数量</label>
                    <div style="padding: 8px 0;">${skill.script_count}</div>
                </div>
            </div>
        </div>

        <div class="form-section">
            <div class="form-section-title">📄 SKILL.md 内容</div>
            <textarea class="form-textarea" readonly style="min-height: 200px; background: #1e1e1e; color: #d4d4d4;">${escapeHtml(data.skill_md || '')}</textarea>
        </div>

        ${scriptsHtml}
    `;

    // 强制设置白色背景
    const modalInner = modal.querySelector('.modal');
    if (modalInner) {
        modalInner.style.backgroundColor = '#ffffff';
        const header = modalInner.querySelector('.modal-header');
        const footer = modalInner.querySelector('.modal-footer');
        if (header) header.style.backgroundColor = '#ffffff';
        if (footer) footer.style.backgroundColor = '#ffffff';
    }
    body.style.backgroundColor = '#ffffff';

    modal.classList.add('active');
}

function closeViewModal() {
    const modal = document.getElementById('viewModal');
    modal.classList.remove('active');
    currentSkillId = null;
}

function editCurrentSkill() {
    if (currentSkillId) {
        closeViewModal();
        openSkillModal(currentSkillId);
    }
}

async function exportCurrentSkill() {
    if (!currentSkillId) return;

    try {
        // 🆕 使用 authFetch
        const response = await authSkillFetch(`${getSkillApiUrl()}/skills/${currentSkillId}/export`);
        if (!response) return;

        const data = await response.json();

        if (data.success) {
            // 创建下载
            const exportData = {
                skill_md: data.skill_md,
                scripts: data.scripts,
                exported_at: data.exported_at
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentSkillId}-skill-export.json`;
            a.click();
            URL.revokeObjectURL(url);

            showToast('Skill 导出成功', 'success');
        } else {
            showToast('导出失败', 'error');
        }
    } catch (error) {
        console.error('导出失败:', error);
        showToast('导出请求失败', 'error');
    }
}

function confirmDelete(skillId) {
    currentSkillId = skillId;
    const modal = document.getElementById('confirmModal');
    const skill = skills.find(s => s.id === skillId);

    document.getElementById('confirmMessage').textContent =
        `确定要删除 "${skill?.name || skillId}" 吗？此操作不可恢复。`;

    document.getElementById('confirmDeleteBtn').onclick = () => deleteSkill(skillId);

    // 强制设置白色背景
    const modalInner = modal.querySelector('.modal');
    if (modalInner) {
        modalInner.style.backgroundColor = '#ffffff';
        const body = modalInner.querySelector('.modal-body');
        if (body) body.style.backgroundColor = '#ffffff';
    }

    modal.classList.add('active');
}

function closeConfirmModal() {
    const modal = document.getElementById('confirmModal');
    modal.classList.remove('active');
    currentSkillId = null;
}

async function deleteSkill(skillId) {
    try {
        // 🆕 使用 authFetch
        const response = await authSkillFetch(`${getSkillApiUrl()}/skills/${skillId}`, {
            method: 'DELETE'
        });
        if (!response) return;

        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');
            closeConfirmModal();
            await loadSkills();
        } else {
            showToast(result.message || '删除失败', 'error');
        }
    } catch (error) {
        console.error('删除失败:', error);
        showToast('删除请求失败', 'error');
    }
}

// ==================== 工具函数 ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 暴露给全局
window.viewSkill = viewSkill;
window.openSkillModal = openSkillModal;
window.closeSkillModal = closeSkillModal;
window.addScriptField = addScriptField;
window.removeScriptField = removeScriptField;
window.validateSkill = validateSkill;
window.installSkill = installSkill;
window.confirmDelete = confirmDelete;
window.closeConfirmModal = closeConfirmModal;
window.deleteCurrentSkill = deleteSkill;
window.editCurrentSkill = editCurrentSkill;
window.exportCurrentSkill = exportCurrentSkill;
window.closeViewModal = closeViewModal;
