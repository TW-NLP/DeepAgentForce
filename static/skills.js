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
            window.location.href = '/login.html';
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

    // 兼容 skills.html 和 index.html 的元素 ID
    const statTotalEl = document.getElementById('statTotal') || document.getElementById('skillCount');
    const statScriptsEl = document.getElementById('statScripts');

    if (statTotalEl) statTotalEl.textContent = total;
    if (statScriptsEl) statScriptsEl.textContent = totalScripts;
}

// ==================== 渲染 Skills ====================

function renderSkills() {
    var grid = document.getElementById('skillsGrid');
    if (!grid) return;
    grid.innerHTML = '';

    // 添加"添加技能"卡片
    var addCard = document.createElement('div');
    addCard.className = 'sp-add-card';
    addCard.onclick = function() { openSkillModal(); };
    addCard.innerHTML = '<div class="sp-add-icon">+</div><div class="sp-add-title">添加新技能</div><div class="sp-add-hint">上传自定义技能插件</div>';
    grid.appendChild(addCard);

    // 更新计数
    var count = document.getElementById('skillCount');
    var countBadge = document.getElementById('skillCountBadge');
    var scriptCount = document.getElementById('skillScriptCount');
    if (count) count.textContent = skills.length;
    if (countBadge) countBadge.textContent = skills.length;
    if (scriptCount) {
        var custom = skills.filter(function(s) {
            return ['pdf-processing', 'rag-query', 'web-search'].indexOf(s.id) === -1;
        }).length;
        scriptCount.textContent = custom;
    }

    // 渲染每个 Skill 卡片
    skills.forEach(function(skill) {
        var card = createSkillCard(skill);
        grid.appendChild(card);
    });
}

function createSkillCard(skill) {
    var card = document.createElement('div');
    card.className = 'sp-card';

    var isBuiltIn = ['pdf-processing', 'rag-query', 'web-search'].indexOf(skill.id) !== -1;
    var iconClass = getIconClass(skill.id);
    var iconText = getIconText(skill.id);

    var badge = isBuiltIn
        ? '<span class="sp-badge sp-badge-builtin">内置</span>'
        : '<span class="sp-badge sp-badge-custom">自定义</span>';

    var deleteBtn = !isBuiltIn
        ? '<button class="sp-action-btn danger" onclick="confirmDelete(\'' + skill.id + '\')" title="删除"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>'
        : '';

    card.innerHTML =
        '<div class="sp-card-top">' +
            '<div class="sp-card-icon ' + iconClass + '">' + iconText + '</div>' +
            '<div>' +
                '<div class="sp-card-name">' + skill.name + ' ' + badge + '</div>' +
                '<div class="sp-card-ver">v' + skill.version + ' · ' + (skill.author || '未知作者') + '</div>' +
            '</div>' +
        '</div>' +
        '<div class="sp-card-desc">' + (skill.description || '暂无描述') + '</div>' +
        (skill.tags && skill.tags.length > 0
            ? '<div class="sp-card-tags">' + skill.tags.map(function(t) { return '<span class="sp-card-tag">' + t + '</span>'; }).join('') + '</div>'
            : '') +
        '<div class="sp-card-bottom">' +
            '<div class="sp-card-meta">' + (skill.script_count || 0) + ' 个脚本</div>' +
            '<div class="sp-card-actions">' +
                '<button class="sp-action-btn view" onclick="viewSkill(\'' + skill.id + '\')" title="查看"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>' +
                (!isBuiltIn ? '<button class="sp-action-btn edit" onclick="editSkill(\'' + skill.id + '\')" title="编辑"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>' : '') +
                deleteBtn +
            '</div>' +
        '</div>';

    return card;
}

function getIconClass(skillId) {
    var map = {
        'pdf-processing': 'blue',
        'rag-query': 'green',
        'web-search': 'orange'
    };
    return map[skillId] || 'default';
}

function getIconText(skillId) {
    var map = {
        'pdf-processing': 'PDF',
        'rag-query': 'RAG',
        'web-search': 'WEB'
    };
    return map[skillId] || 'SK';
}

function getSkillIcon(skillId) {
    return getIconText(skillId);
}

// ==================== Modal 操作 ====================

function openSkillModal(skillId = null) {
    const modal = document.getElementById('skillModal');
    const title = document.getElementById('modalTitle');
    const form = document.getElementById('skillForm');

    if (!modal) {
        console.error('skillModal not found');
        return;
    }

    // 重置表单
    if (form) form.reset();
    scriptFields = [];

    const scriptListEl = document.getElementById('scriptList');
    const validationResultsEl = document.getElementById('validationResults');
    if (scriptListEl) scriptListEl.innerHTML = '';
    if (validationResultsEl) validationResultsEl.classList.remove('show');

    if (skillId) {
        // 编辑模式
        isEditMode = true;
        currentSkillId = skillId;
        if (title) title.textContent = '编辑 Skill';

        const skill = skills.find(s => s.id === skillId);
        if (skill) {
            loadSkillForEdit(skill);
        }
    } else {
        // 添加模式
        isEditMode = false;
        currentSkillId = null;
        if (title) title.textContent = '添加新 Skill';

        // 添加默认脚本字段
        addScriptField();
    }

    modal.classList.add('active');
}

function closeSkillModal() {
    const modal = document.getElementById('skillModal');
    if (modal) modal.classList.remove('active');
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
    if (!container) {
        console.error('scriptList not found');
        return;
    }
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
    if (!container) return;

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
    var modal = document.getElementById('viewModal');
    var title = document.getElementById('viewModalTitle');
    var body = document.getElementById('viewModalBody');

    if (!modal || !title || !body) {
        console.error('ViewModal elements not found');
        return;
    }

    var skill = skills.find(function(s) { return s.id === skillId; });
    if (!skill) return;

    var isBuiltIn = ['pdf-processing', 'rag-query', 'web-search'].indexOf(skill.id) !== -1;
    var iconClass = getIconClass(skill.id);
    var iconText = getIconText(skill.id);
    var badgeText = isBuiltIn ? '内置' : '自定义';

    title.textContent = '查看技能';

    body.innerHTML =
        '<div class="detail-header">' +
            '<div class="detail-icon ' + iconClass + '">' + iconText + '</div>' +
            '<div class="detail-title">' +
                '<h3>' + skill.name + '</h3>' +
                '<span class="detail-type">' + badgeText + '</span>' +
            '</div>' +
        '</div>' +
        '<div class="detail-desc">' + (skill.description || '暂无描述') + '</div>' +
        '<div class="detail-meta">' +
            '<div class="detail-meta-item">' +
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
                'v' + skill.version +
            '</div>' +
            '<div class="detail-meta-item">' +
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>' +
                (skill.author || '未知作者') +
            '</div>' +
            '<div class="detail-meta-item">' +
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
                (skill.script_count || 0) + ' 个脚本' +
            '</div>' +
        '</div>';

    // 控制脚注按钮显示
    var footer = modal.querySelector('.modal-footer');
    if (footer) {
        var editBtn = footer.querySelector('[onclick="editCurrentSkill()"]');
        var deleteBtn = footer.querySelector('[onclick="deleteCurrentSkill()"]');
        if (editBtn) editBtn.style.display = isBuiltIn ? 'none' : '';
        if (deleteBtn) deleteBtn.style.display = isBuiltIn ? 'none' : '';
    }

    modal.classList.add('active');
}

function closeViewModal() {
    const modal = document.getElementById('viewModal');
    if (modal) modal.classList.remove('active');
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

    const confirmMessage = document.getElementById('confirmMessage');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

    if (confirmMessage) {
        confirmMessage.textContent = `确定要删除 "${skill?.name || skillId}" 吗？此操作不可恢复。`;
    }
    if (confirmDeleteBtn) {
        confirmDeleteBtn.onclick = () => deleteSkill(skillId);
    }

    // 强制设置背景色
    const modalInner = modal?.querySelector('.modal');
    if (modalInner) {
        modalInner.style.removeProperty('background-color');
        const body = modalInner.querySelector('.modal-body');
        if (body) body.style.removeProperty('background-color');
    }

    if (modal) modal.classList.add('active');
}

function closeConfirmModal() {
    const modal = document.getElementById('confirmModal');
    if (modal) modal.classList.remove('active');
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
// 直接打开编辑弹窗
function editSkill(skillId) {
    openSkillModal(skillId);
}

window.openSkillModal = openSkillModal;
window.editSkill = editSkill;
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