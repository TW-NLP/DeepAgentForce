/**
 * Skill 管理 JavaScript
 * 处理 Skill 的增删改查、导入导出等功能
 */

(() => {

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
let currentSkillFilter = 'all';
let currentSkillSort = 'updated';
let skillSearchKeyword = '';
let currentDrawerSkillId = null;
let currentDrawerContent = null;
let currentDrawerLoading = false;

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
    const builtinCount = skills.filter(function(s) { return isBuiltInSkill(s); }).length;
    const customCount = total - builtinCount;

    // 兼容 skills.html 和 index.html 的元素 ID
    const statTotalEl = document.getElementById('statTotal') || document.getElementById('skillCount');
    const statScriptsEl = document.getElementById('statScripts') || document.getElementById('skillScriptCount');
    const statBuiltinEl = document.getElementById('builtinSkillCount');
    const statCustomEl = document.getElementById('customSkillCount');

    if (statTotalEl) statTotalEl.textContent = total;
    if (statScriptsEl) statScriptsEl.textContent = totalScripts;
    if (statBuiltinEl) statBuiltinEl.textContent = builtinCount;
    if (statCustomEl) statCustomEl.textContent = customCount;
}

// ==================== 渲染 Skills ====================

function renderSkills() {
    var list = document.getElementById('skillsGrid');
    var emptyState = document.getElementById('skillsEmptyState');
    if (!list) return;

    var visibleSkills = getVisibleSkills();

    var count = document.getElementById('skillCount');
    var countBadge = document.getElementById('skillCountBadge');
    var hint = document.getElementById('skillListHint');
    if (count) count.textContent = skills.length;
    if (countBadge) countBadge.textContent = visibleSkills.length;
    if (hint) {
        hint.textContent = skillSearchKeyword || currentSkillFilter !== 'all'
            ? ('当前显示 ' + visibleSkills.length + ' 个技能')
            : '点击任意技能查看详情';
    }

    list.innerHTML = '';

    if (visibleSkills.length === 0) {
        if (emptyState) emptyState.style.display = 'flex';
        if (currentDrawerSkillId && !skills.find(function(skill) { return skill.id === currentDrawerSkillId; })) {
            closeSkillDrawer();
        }
        return;
    }

    if (emptyState) emptyState.style.display = 'none';

    visibleSkills.forEach(function(skill) {
        var row = createSkillCard(skill);
        list.appendChild(row);
    });

    if (currentDrawerSkillId) {
        var selected = skills.find(function(skill) { return skill.id === currentDrawerSkillId; });
        if (selected) {
            document.querySelectorAll('.skill-row').forEach(function(row) {
                row.classList.toggle('active', row.getAttribute('data-skill-id') === currentDrawerSkillId);
            });
        } else {
            closeSkillDrawer();
        }
    }
}

function createSkillCard(skill) {
    var card = document.createElement('div');
    card.className = 'skill-row';
    card.setAttribute('data-skill-id', skill.id);
    card.onclick = function() { openSkillDrawer(skill.id, { loading: true }); loadSkillDrawerContent(skill.id); };

    var isBuiltIn = isBuiltInSkill(skill);
    var iconClass = getIconClass(skill.id);
    var iconText = getIconText(skill.id);

    var badge = isBuiltIn
        ? '<span class="sp-badge sp-badge-builtin">内置</span>'
        : '<span class="sp-badge sp-badge-custom">自定义</span>';

    var tags = (skill.tags || []).slice(0, 4).map(function(t) {
        return '<span class="skill-row-tag">' + escapeHtml(t) + '</span>';
    }).join('');

    var scriptNames = (skill.script_names || []).slice(0, 3).map(function(name) {
        return '<span class="skill-chip muted">' + escapeHtml(name) + '</span>';
    }).join('');

    card.innerHTML =
        '<div class="skill-row-icon ' + iconClass + '">' + escapeHtml(iconText) + '</div>' +
        '<div class="skill-row-main">' +
            '<div class="skill-row-topline">' +
                '<div class="skill-row-title">' +
                    '<div class="skill-row-name">' + escapeHtml(skill.name) + '</div>' +
                    badge +
                '</div>' +
                '<div class="skill-row-actions">' +
                    '<button class="sp-action-btn" onclick="event.stopPropagation(); viewSkill(\'' + skill.id + '\')" title="查看"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>' +
                    (!isBuiltIn ? '<button class="sp-action-btn edit" onclick="event.stopPropagation(); editSkill(\'' + skill.id + '\')" title="编辑"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>' : '') +
                    (!isBuiltIn ? '<button class="sp-action-btn danger" onclick="event.stopPropagation(); confirmDelete(\'' + skill.id + '\')" title="删除"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>' : '') +
                '</div>' +
            '</div>' +
            '<div class="skill-row-subtitle">' + escapeHtml(skill.description || skill.summary || '暂无描述') + '</div>' +
            '<div class="skill-row-meta">' +
                '<span class="skill-chip">' + (skill.script_count || 0) + ' 个脚本</span>' +
                '<span class="skill-chip muted">' + escapeHtml(skill.author || 'Unknown') + '</span>' +
                '<span class="skill-chip muted">' + escapeHtml(formatRelativeTime(skill.modified_at || skill.created_at)) + '</span>' +
                '<span class="skill-chip muted">' + escapeHtml(formatFileSize(skill.size_bytes || 0)) + '</span>' +
            '</div>' +
            (scriptNames ? '<div class="skill-row-meta">' + scriptNames + '</div>' : '') +
            (tags ? '<div class="skill-row-tags">' + tags + '</div>' : '') +
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

function formatRelativeTime(dateString) {
    if (!dateString) return '刚刚';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) return '未知时间';
    const diff = Date.now() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return minutes + ' 分钟前';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + ' 小时前';
    const days = Math.floor(hours / 24);
    if (days < 30) return days + ' 天前';
    return new Date(dateString).toLocaleDateString('zh-CN');
}

function formatFileSize(bytes) {
    var size = Number(bytes) || 0;
    if (size <= 0) return '0 B';
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var idx = 0;
    while (size >= 1024 && idx < units.length - 1) {
        size /= 1024;
        idx += 1;
    }
    var value = size >= 10 || idx === 0 ? size.toFixed(0) : size.toFixed(1);
    return value + ' ' + units[idx];
}

function isBuiltInSkill(skill) {
    return ['pdf-processing', 'rag-query', 'web-search'].indexOf(skill.id) !== -1 || skill.source === 'builtin' || skill.is_builtin;
}

function getVisibleSkills() {
    var keyword = (skillSearchKeyword || '').trim().toLowerCase();
    return skills.filter(function(skill) {
        var haystack = [
            skill.name,
            skill.description,
            skill.author,
            skill.summary,
            (skill.tags || []).join(' '),
            (skill.script_names || []).join(' ')
        ].join(' ').toLowerCase();
        var matchesKeyword = !keyword || haystack.indexOf(keyword) !== -1;
        var builtin = isBuiltInSkill(skill);
        var matchesFilter = currentSkillFilter === 'all' ||
            (currentSkillFilter === 'builtin' && builtin) ||
            (currentSkillFilter === 'custom' && !builtin);
        return matchesKeyword && matchesFilter;
    }).sort(function(a, b) {
        if (currentSkillSort === 'name') {
            return (a.name || '').localeCompare(b.name || '', 'zh-Hans-CN');
        }
        if (currentSkillSort === 'scripts') {
            return (b.script_count || 0) - (a.script_count || 0);
        }
        var at = new Date(a.modified_at || a.created_at || 0).getTime();
        var bt = new Date(b.modified_at || b.created_at || 0).getTime();
        return bt - at;
    });
}

function getDrawerElements() {
    return {
        drawer: document.getElementById('skillDetailDrawer'),
        empty: document.getElementById('skillDrawerEmpty'),
        shell: document.getElementById('skillDrawerShell'),
        title: document.getElementById('skillDrawerTitle'),
        subtitle: document.getElementById('skillDrawerSubtitle'),
        icon: document.getElementById('skillDrawerIcon'),
        status: document.getElementById('skillDrawerStatus'),
        meta: document.getElementById('skillDrawerMeta'),
        description: document.getElementById('skillDrawerDescription'),
        markdown: document.getElementById('skillDrawerMarkdown'),
        scripts: document.getElementById('skillDrawerScripts'),
        actions: document.getElementById('skillDrawerActions')
    };
}

function hasSkillDrawer() {
    return !!document.getElementById('skillDetailDrawer');
}

function hasSkillViewModal() {
    return !!document.getElementById('viewModal');
}

function renderDrawer(skill, content, loading) {
    var els = getDrawerElements();
    if (!els.drawer || !els.empty || !els.shell) return;

    if (!skill) {
        els.empty.style.display = 'flex';
        els.shell.style.display = 'none';
        els.drawer.classList.remove('has-selection');
        currentDrawerSkillId = null;
        currentDrawerContent = null;
        return;
    }

    els.empty.style.display = 'none';
    els.shell.style.display = 'flex';
    els.drawer.classList.add('has-selection');

    var builtin = isBuiltInSkill(skill);
    if (els.title) els.title.textContent = skill.name || skill.id;
    if (els.subtitle) {
        els.subtitle.textContent = (skill.description || skill.summary || '暂无描述') +
            ' · ' + (skill.author || '未知作者');
    }
    if (els.icon) {
        els.icon.textContent = getSkillIcon(skill.id);
        els.icon.className = 'skill-drawer-icon ' + getIconClass(skill.id);
    }
    if (els.status) {
        els.status.textContent = loading ? '加载中' : (builtin ? '内置技能' : '自定义技能');
    }

    if (els.meta) {
        var tags = (skill.tags || []).map(function(tag) {
            return '<span class="skill-chip">' + escapeHtml(tag) + '</span>';
        }).join('');
        els.meta.innerHTML = [
            '<div class="skill-meta-item"><span class="skill-meta-label">技能 ID</span><span class="skill-meta-value mono">' + escapeHtml(skill.id) + '</span></div>',
            '<div class="skill-meta-item"><span class="skill-meta-label">版本</span><span class="skill-meta-value">' + escapeHtml(skill.version || '1.0.0') + '</span></div>',
            '<div class="skill-meta-item"><span class="skill-meta-label">作者</span><span class="skill-meta-value">' + escapeHtml(skill.author || 'Unknown') + '</span></div>',
            '<div class="skill-meta-item"><span class="skill-meta-label">脚本数量</span><span class="skill-meta-value">' + (skill.script_count || 0) + '</span></div>',
            '<div class="skill-meta-item"><span class="skill-meta-label">更新时间</span><span class="skill-meta-value">' + escapeHtml(formatRelativeTime(skill.modified_at || skill.created_at)) + '</span></div>',
            '<div class="skill-meta-item"><span class="skill-meta-label">大小</span><span class="skill-meta-value">' + escapeHtml(formatFileSize(skill.size_bytes || 0)) + '</span></div>'
        ].join('');
        if (tags) {
            els.meta.innerHTML += '<div class="skill-meta-item" style="grid-column: 1 / -1;"><span class="skill-meta-label">标签</span><div class="sp-filter-btns" style="margin-top: 0;">' + tags + '</div></div>';
        }
    }

    if (els.description) {
        els.description.textContent = skill.summary || skill.description || '暂无描述';
    }

    if (els.markdown) {
        if (loading || !content) {
            els.markdown.textContent = loading ? '正在加载 SKILL.md ...' : '点击“查看”或“预览内容”加载 Markdown。';
        } else {
            els.markdown.textContent = content.skill_md || '暂无 SKILL.md 内容';
        }
    }

    if (els.scripts) {
        if (loading || !content) {
            els.scripts.innerHTML = '<div class="skill-drawer-script">脚本内容加载中...</div>';
        } else {
            var scripts = content.scripts || {};
            var scriptEntries = Object.entries(scripts);
            els.scripts.innerHTML = scriptEntries.length ? scriptEntries.map(function(entry) {
                var name = entry[0];
                var scriptContent = entry[1] || '';
                return '<div class="skill-drawer-script">' +
                    '<div class="skill-drawer-script-name">' + escapeHtml(name) + '</div>' +
                    '<div class="skill-drawer-script-path">' + escapeHtml(scriptContent.substring(0, 220) || '空脚本') + '</div>' +
                '</div>';
            }).join('') : '<div class="skill-drawer-script">暂无脚本文件</div>';
        }
    }

    if (els.actions) {
        var exportBtn = '<button class="skill-drawer-btn secondary" type="button" onclick="exportCurrentSkill()"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M5 21h14"/></svg>导出</button>';
        var editBtn = builtin ? '' : '<button class="skill-drawer-btn secondary" type="button" onclick="editSkill(\'' + skill.id + '\')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>编辑</button>';
        var deleteBtn = builtin ? '' : '<button class="skill-drawer-btn danger" type="button" onclick="confirmDelete(\'' + skill.id + '\')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>删除</button>';
        els.actions.innerHTML = '<button class="skill-drawer-btn" type="button" onclick="viewSkill(\'' + skill.id + '\')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>查看内容</button>' + editBtn + exportBtn + deleteBtn;
    }
}

function openSkillDrawer(skillId, options) {
    var skill = skills.find(function(item) { return item.id === skillId; });
    if (!skill) {
        renderDrawer(null);
        return;
    }

    currentSkillId = skillId;
    currentDrawerSkillId = skillId;
    currentDrawerLoading = options && options.loading;
    renderDrawer(skill, null, true);
    document.querySelectorAll('.skill-row').forEach(function(row) {
        row.classList.toggle('active', row.getAttribute('data-skill-id') === skillId);
    });
}

function closeSkillDrawer() {
    currentSkillId = null;
    currentDrawerSkillId = null;
    currentDrawerContent = null;
    currentDrawerLoading = false;
    renderDrawer(null);
    document.querySelectorAll('.skill-row').forEach(function(row) {
        row.classList.remove('active');
    });
}

async function loadSkillDrawerContent(skillId) {
    var skill = skills.find(function(item) { return item.id === skillId; });
    if (!skill) return;

    currentDrawerLoading = true;
    renderDrawer(skill, null, true);

    try {
        const response = await authSkillFetch(getSkillApiUrl() + '/skills/' + skillId + '/content');
        if (!response) return;
        const data = await response.json();
        if (data.success && currentDrawerSkillId === skillId) {
            currentDrawerContent = data;
            renderDrawer(skill, data, false);
        }
    } catch (error) {
        if (currentDrawerSkillId === skillId) {
            renderDrawer(skill, null, false);
        }
        showToast('加载 Skill 内容失败', 'error');
    } finally {
        currentDrawerLoading = false;
    }
}

function setupSkillPageControls() {
    var refreshBtn = document.getElementById('refreshSkillsBtn');
    if (refreshBtn && !refreshBtn.dataset.bound) {
        refreshBtn.dataset.bound = '1';
        refreshBtn.addEventListener('click', function() {
            loadSkills();
        });
    }

    var searchInput = document.getElementById('skillSearchInput');
    if (searchInput && !searchInput.dataset.bound) {
        searchInput.dataset.bound = '1';
        searchInput.addEventListener('input', function() {
            skillSearchKeyword = searchInput.value || '';
            renderSkills();
        });
    }

    var sortSelect = document.getElementById('skillSortSelect');
    if (sortSelect && !sortSelect.dataset.bound) {
        sortSelect.dataset.bound = '1';
        sortSelect.addEventListener('change', function() {
            currentSkillSort = sortSelect.value || 'updated';
            renderSkills();
        });
    }

    document.querySelectorAll('[data-skill-filter]').forEach(function(btn) {
        if (btn.dataset.bound) return;
        btn.dataset.bound = '1';
        btn.addEventListener('click', function() {
            var filter = btn.getAttribute('data-skill-filter') || 'all';
            currentSkillFilter = filter;
            document.querySelectorAll('[data-skill-filter]').forEach(function(item) {
                item.classList.toggle('active', item.getAttribute('data-skill-filter') === filter);
            });
            renderSkills();
        });
    });

    var closeDrawerBtn = document.getElementById('skillDrawerCloseBtn');
    if (closeDrawerBtn && !closeDrawerBtn.dataset.bound) {
        closeDrawerBtn.dataset.bound = '1';
        closeDrawerBtn.addEventListener('click', function() {
            closeSkillDrawer();
        });
    }
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
            if (hasSkillDrawer()) {
                openSkillDrawer(skillId, { loading: false });
                currentDrawerContent = data;
                renderDrawer(data.skill || skills.find(function(item) { return item.id === skillId; }), data, false);
            } else {
                openViewModal(skillId, data);
            }
        } else {
            showToast('加载 Skill 详情失败', 'error');
        }
    } catch (error) {
        console.error('加载 Skill 详情失败:', error);
        showToast('无法加载 Skill 详情', 'error');
    }
}

function openViewModal(skillId, data) {
    if (hasSkillDrawer()) {
        var skill = data?.skill || skills.find(function(s) { return s.id === skillId; });
        if (skill) {
            currentSkillId = skillId;
            currentDrawerSkillId = skillId;
            currentDrawerContent = data || null;
            renderDrawer(skill, data || null, false);
            document.querySelectorAll('.skill-row').forEach(function(row) {
                row.classList.toggle('active', row.getAttribute('data-skill-id') === skillId);
            });
        }
        return;
    }

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
    if (hasSkillDrawer()) {
        closeSkillDrawer();
        return;
    }

    const modal = document.getElementById('viewModal');
    if (modal) modal.classList.remove('active');
    currentSkillId = null;
}

function editCurrentSkill() {
    if (currentSkillId) {
        const skillId = currentSkillId;
        closeViewModal();
        openSkillModal(skillId);
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
window.loadSkills = loadSkills;
// 直接打开编辑弹窗
function editSkill(skillId) {
    openSkillModal(skillId);
}

window.openSkillDrawer = openSkillDrawer;
window.closeSkillDrawer = closeSkillDrawer;
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

document.addEventListener('DOMContentLoaded', setupSkillPageControls);

})();
