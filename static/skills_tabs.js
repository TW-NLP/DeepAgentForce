/**
 * 技能管理页 - 子标签（技能 / 工具 / MCP）
 * 复用 index.html 原生 sp-* / skill-row / modal 设计系统与全局 showToast / getApiUrl。
 * 工具 = 内置(只读) + MCP(只读) + 自定义 Python(增删改查，子进程沙箱执行)。
 * 多租户：所有请求带 Bearer token，后端按 tenant_uuid 隔离。
 */
(() => {

    // ==================== 通用工具 ====================
    function apiBase() {
        if (window.getApiUrl) return window.getApiUrl();
        return `${window.location.origin}/api`;
    }

    async function authFetch(url, options = {}) {
        const token = localStorage.getItem('access_token');
        const headers = { ...(options.headers || {}) };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        let resp = await fetch(url, { ...options, headers });
        if (resp.status === 401 && token) {
            const refreshed = await window.auth?.refreshAccessToken?.();
            if (refreshed) {
                headers['Authorization'] = `Bearer ${localStorage.getItem('access_token')}`;
                resp = await fetch(url, { ...options, headers });
            } else {
                window.location.href = '/login.html';
                return null;
            }
        }
        return resp;
    }

    function esc(s) {
        const div = document.createElement('div');
        div.textContent = s == null ? '' : String(s);
        return div.innerHTML;
    }

    function toast(msg, type = 'success') {
        (window.showToast || ((m) => alert(m)))(msg, type);
    }

    function fmtSize(bytes) {
        let size = Number(bytes) || 0;
        if (size <= 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        let i = 0;
        while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
        return (size >= 10 || i === 0 ? size.toFixed(0) : size.toFixed(1)) + ' ' + units[i];
    }

    // 通用卡片（沿用 skill-row 骨架）
    function card({ icon, iconClass, name, badge, subtitle, metas, tags, actions }) {
        const metaHtml = (metas || []).map(m => `<span class="skill-chip muted">${m}</span>`).join('');
        const tagHtml = (tags || []).map(t => `<span class="skill-row-tag">${esc(t)}</span>`).join('');
        return `
        <div class="skill-row" style="cursor:default">
            <div class="skill-row-icon ${iconClass || 'default'}">${esc(icon || 'T')}</div>
            <div class="skill-row-main">
                <div class="skill-row-topline">
                    <div class="skill-row-title">
                        <div class="skill-row-name">${esc(name)}</div>
                        ${badge || ''}
                    </div>
                    <div class="skill-row-actions">${actions || ''}</div>
                </div>
                <div class="skill-row-subtitle">${esc(subtitle || '')}</div>
                ${metaHtml ? `<div class="skill-row-meta">${metaHtml}</div>` : ''}
                ${tagHtml ? `<div class="skill-row-tags">${tagHtml}</div>` : ''}
            </div>
        </div>`;
    }

    // ==================== 子标签切换 ====================
    const SUBTITLES = {
        skills: '管理智能体的技能插件，扩展系统能力',
        tools: '内置与 MCP 工具只读展示；自定义 Python 工具在受限子进程沙箱中执行',
        mcp: '连接 MCP Server，把外部工具接入智能体（渐进式披露）',
    };
    const loaded = { tools: false, mcp: false };
    let currentSubtab = 'skills';

    function switchSubtab(tab) {
        currentSubtab = tab;
        document.querySelectorAll('.sp-subtab').forEach(b =>
            b.classList.toggle('active', b.getAttribute('data-subtab') === tab));
        document.querySelectorAll('.sp-subpanel').forEach(p =>
            p.classList.toggle('active', p.getAttribute('data-subpanel') === tab));
        document.querySelectorAll('.sp-actions-group').forEach(g =>
            g.style.display = (g.getAttribute('data-actions') === tab) ? 'inline-flex' : 'none');
        const sub = document.getElementById('spSubtitle');
        if (sub && SUBTITLES[tab]) sub.textContent = SUBTITLES[tab];

        if (tab === 'tools' && !loaded.tools) { loaded.tools = true; loadCustomTools(); }
        if (tab === 'mcp' && !loaded.mcp) { loaded.mcp = true; loadMcpServers(); }
    }

    function bindSubtabs() {
        const bar = document.getElementById('spSubtabs');
        if (!bar || bar.dataset.bound) return;
        bar.dataset.bound = '1';
        bar.addEventListener('click', (e) => {
            const btn = e.target.closest('.sp-subtab');
            if (!btn) return;
            switchSubtab(btn.getAttribute('data-subtab'));
        });
    }

    // ==================== 工具：加载与渲染 ====================
    async function loadCustomTools() {
        try {
            const resp = await authFetch(`${apiBase()}/tools`);
            if (!resp) return;
            const data = await resp.json();
            if (!data.success) { toast('加载工具失败', 'error'); return; }

            renderReadonlyTools(data.builtin || [], data.mcp || []);
            renderCustomTools(data.custom || []);

            setText('statBuiltinTools', (data.builtin || []).length);
            setText('statMcpTools', (data.mcp || []).length);
            setText('statCustomTools', (data.custom || []).length);
            setText('customToolsCount', (data.custom || []).length);
        } catch (e) {
            console.error(e);
            toast('加载工具异常: ' + e.message, 'error');
        }
    }

    function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

    function renderReadonlyTools(builtin, mcp) {
        const grid = document.getElementById('readonlyToolsGrid');
        if (!grid) return;
        const b = builtin.map(t => card({
            icon: 'B', iconClass: 'green', name: t.name,
            badge: '<span class="sp-badge tool-builtin">' + esc(t.category_label || '内置') + '</span>',
            subtitle: t.description,
        }));
        const m = mcp.map(t => card({
            icon: 'M', iconClass: 'blue', name: t.name,
            badge: '<span class="sp-badge tool-mcp">' + esc(t.category_label || 'MCP') + '</span>',
            subtitle: t.description,
        }));
        grid.innerHTML = b.concat(m).join('') ||
            '<div class="sp-empty-state" style="display:flex"><div class="sp-empty-icon">🧰</div><div class="sp-empty-text">暂无工具</div></div>';
    }

    function renderCustomTools(items) {
        const grid = document.getElementById('customToolsGrid');
        const empty = document.getElementById('customToolsEmpty');
        if (!grid) return;
        if (!items.length) {
            grid.innerHTML = '';
            if (empty) empty.style.display = 'flex';
            return;
        }
        if (empty) empty.style.display = 'none';
        grid.innerHTML = items.map(it => {
            const toolNames = (it.tools || []).map(t => t.name);
            const subtitle = it.error
                ? '加载失败：' + it.error
                : (it.tools || []).map(t => t.name + '：' + (t.description || '')).join('；') || '无可用工具';
            const metas = [
                (it.tools || []).length + ' 个工具',
                fmtSize(it.size_bytes),
                esc(it.modified_at || ''),
            ];
            const actions =
                `<button class="sp-action-btn edit" title="编辑" onclick="editCustomTool('${esc(it.tool_id)}')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>` +
                `<button class="sp-action-btn danger" title="删除" onclick="deleteCustomTool('${esc(it.tool_id)}')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg></button>`;
            return card({
                icon: 'PY', iconClass: 'orange', name: it.tool_id,
                badge: '<span class="sp-badge tool-custom">自定义</span>',
                subtitle, metas, tags: toolNames, actions,
            });
        }).join('');
    }

    // ==================== 工具：弹窗 CRUD ====================
    let editingToolId = null;

    function openToolModal() {
        editingToolId = null;
        document.getElementById('toolModalTitle').textContent = '添加自定义工具';
        const idInput = document.getElementById('toolId');
        idInput.value = '';
        idInput.removeAttribute('readonly');
        document.getElementById('toolCode').value = '';
        document.getElementById('toolValidateResult').innerHTML = '';
        document.getElementById('toolModal').classList.add('active');
    }
    function closeToolModal() { document.getElementById('toolModal').classList.remove('active'); }

    async function editCustomTool(toolId) {
        const resp = await authFetch(`${apiBase()}/tools/custom/${encodeURIComponent(toolId)}`);
        if (!resp || !resp.ok) { toast('获取源码失败', 'error'); return; }
        const data = await resp.json();
        editingToolId = toolId;
        document.getElementById('toolModalTitle').textContent = `编辑：${toolId}`;
        const idInput = document.getElementById('toolId');
        idInput.value = toolId;
        idInput.setAttribute('readonly', 'readonly');
        document.getElementById('toolCode').value = data.code || '';
        document.getElementById('toolValidateResult').innerHTML = '';
        document.getElementById('toolModal').classList.add('active');
    }

    async function loadToolTemplate() {
        const resp = await authFetch(`${apiBase()}/tools/template`);
        if (!resp) return;
        const data = await resp.json();
        document.getElementById('toolCode').value = data.code || '';
    }

    function renderValidate(elId, data) {
        const el = document.getElementById(elId);
        let html = '';
        html += data.success
            ? `<div class="ok">✓ ${esc(data.message || '校验通过')}</div>`
            : `<div class="err">✕ ${esc(data.message || '校验失败')}</div>`;
        if ((data.tool_names || []).length) html += `<div class="ok">将注册工具：${data.tool_names.map(esc).join('、')}</div>`;
        (data.errors || []).forEach(e => html += `<div class="err">· ${esc(e)}</div>`);
        (data.warnings || []).forEach(w => html += `<div class="warn">⚠ ${esc(w)}</div>`);
        el.innerHTML = html;
    }

    async function validateToolCode() {
        const code = document.getElementById('toolCode').value;
        const resp = await authFetch(`${apiBase()}/tools/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ code }),
        });
        if (!resp) return;
        renderValidate('toolValidateResult', await resp.json());
    }

    async function saveToolCode() {
        const toolId = document.getElementById('toolId').value.trim();
        const code = document.getElementById('toolCode').value;
        if (!toolId) { toast('请填写工具文件名', 'error'); return; }
        if (!code.trim()) { toast('请填写代码', 'error'); return; }

        const force = editingToolId === toolId ? 'true' : 'false';
        let resp = await authFetch(`${apiBase()}/tools/custom`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ tool_id: toolId, code, force }),
        });
        if (!resp) return;
        let data = await resp.json();

        if (!data.success && /已存在/.test(data.message || '') && force === 'false') {
            if (confirm(`工具 '${toolId}' 已存在，是否覆盖？`)) {
                resp = await authFetch(`${apiBase()}/tools/custom`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ tool_id: toolId, code, force: 'true' }),
                });
                data = await resp.json();
            } else return;
        }

        if (data.success) { toast(data.message || '已保存'); closeToolModal(); loadCustomTools(); }
        else { renderValidate('toolValidateResult', data); toast(data.message || '保存失败', 'error'); }
    }

    async function deleteCustomTool(toolId) {
        if (!confirm(`确定删除自定义工具 '${toolId}'？`)) return;
        const resp = await authFetch(`${apiBase()}/tools/custom/${encodeURIComponent(toolId)}`, { method: 'DELETE' });
        if (!resp) return;
        if (resp.ok) { toast('已删除'); loadCustomTools(); }
        else { const d = await resp.json().catch(() => ({})); toast(d.detail || '删除失败', 'error'); }
    }

    // ==================== MCP：加载与渲染 ====================
    let serversCache = [];

    async function loadMcpServers() {
        try {
            const resp = await authFetch(`${apiBase()}/mcp/servers`);
            if (!resp) return;
            const data = await resp.json();
            if (!data.success) { toast('加载 MCP 失败', 'error'); return; }
            serversCache = data.servers || [];
            renderMcpServers(serversCache);
            setText('mcpServersCount', serversCache.length);
        } catch (e) {
            console.error(e);
            toast('加载 MCP 异常: ' + e.message, 'error');
        }
    }

    function renderMcpServers(servers) {
        const grid = document.getElementById('mcpServersGrid');
        const empty = document.getElementById('mcpServersEmpty');
        if (!grid) return;
        if (!servers.length) { grid.innerHTML = ''; if (empty) empty.style.display = 'flex'; return; }
        if (empty) empty.style.display = 'none';

        grid.innerHTML = servers.map(s => {
            const isGlobal = s.source === 'global';
            const target = s.url || ((s.command || '') + ' ' + (s.args || []).join(' ')).trim();
            const badge =
                (isGlobal ? '<span class="sp-badge mcp-global">全局·只读</span>' : '<span class="sp-badge mcp-tenant">我的</span>') +
                (s.enabled ? ' <span class="sp-badge mcp-on">已启用</span>' : ' <span class="sp-badge mcp-off">已禁用</span>');
            let actions = '';
            if (!isGlobal) {
                actions =
                    `<button class="sp-action-btn" title="${s.enabled ? '禁用' : '启用'}" onclick="toggleMcpServer('${esc(s.name)}', ${s.enabled ? 'false' : 'true'})"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg></button>` +
                    `<button class="sp-action-btn edit" title="编辑" onclick="editMcpServer('${esc(s.name)}')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>` +
                    `<button class="sp-action-btn danger" title="删除" onclick="deleteMcpServer('${esc(s.name)}')"><svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg></button>`;
            }
            return card({
                icon: '🔌', iconClass: isGlobal ? 'default' : 'green', name: s.name,
                badge, subtitle: target,
                metas: ['传输：' + (s.transport || 'stdio')],
                actions,
            });
        }).join('');
    }

    // ==================== MCP：弹窗 CRUD ====================
    function gv(id) { const el = document.getElementById(id); return el ? el.value : ''; }
    function sv(id, v) { const el = document.getElementById(id); if (el) el.value = v; }
    function linesToArray(t) { return (t || '').split('\n').map(l => l.trim()).filter(Boolean); }
    function textToKv(t) {
        const out = {};
        (t || '').split('\n').forEach(line => {
            const i = line.indexOf('=');
            if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
        });
        return out;
    }
    function kvToText(o) { return Object.entries(o || {}).map(([k, v]) => `${k}=${v}`).join('\n'); }

    function onMcpTransportChange() {
        const t = gv('mcpTransport');
        document.getElementById('mcpStdioFields').style.display = (t === 'stdio') ? '' : 'none';
        document.getElementById('mcpHttpFields').style.display = (t === 'stdio') ? 'none' : '';
    }

    function openMcpModal() {
        sv('mcpName', '');
        document.getElementById('mcpName').removeAttribute('readonly');
        sv('mcpTransport', 'stdio');
        sv('mcpCommand', ''); sv('mcpArgs', ''); sv('mcpEnv', '');
        sv('mcpUrl', ''); sv('mcpHeaders', '');
        document.getElementById('mcpTestResult').innerHTML = '';
        document.getElementById('mcpModalTitle').textContent = '添加 MCP Server';
        onMcpTransportChange();
        document.getElementById('mcpModal').classList.add('active');
    }
    function closeMcpModal() { document.getElementById('mcpModal').classList.remove('active'); }

    function editMcpServer(name) {
        const s = serversCache.find(x => x.name === name);
        if (!s) { toast('未找到该 server', 'error'); return; }
        sv('mcpName', s.name);
        document.getElementById('mcpName').setAttribute('readonly', 'readonly');
        sv('mcpTransport', s.transport || 'stdio');
        sv('mcpCommand', s.command || '');
        sv('mcpArgs', (s.args || []).join('\n'));
        sv('mcpEnv', kvToText(s.env));
        sv('mcpUrl', s.url || '');
        sv('mcpHeaders', kvToText(s.headers));
        document.getElementById('mcpTestResult').innerHTML = '';
        document.getElementById('mcpModalTitle').textContent = `编辑：${name}`;
        onMcpTransportChange();
        document.getElementById('mcpModal').classList.add('active');
    }

    function buildMcpConfig() {
        const transport = gv('mcpTransport');
        const cfg = { name: gv('mcpName').trim(), transport };
        if (transport === 'stdio') {
            cfg.command = gv('mcpCommand').trim();
            cfg.args = linesToArray(gv('mcpArgs'));
            cfg.env = textToKv(gv('mcpEnv'));
        } else {
            cfg.url = gv('mcpUrl').trim();
            cfg.headers = textToKv(gv('mcpHeaders'));
        }
        return cfg;
    }

    async function testMcpServer() {
        const cfg = buildMcpConfig();
        if (!cfg.name) { toast('请填写名称', 'error'); return; }
        const el = document.getElementById('mcpTestResult');
        el.innerHTML = '<div>正在连接…</div>';
        const resp = await authFetch(`${apiBase()}/mcp/servers/test`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
        });
        if (!resp) return;
        const data = await resp.json();
        if (data.success) {
            const list = (data.tools || []).map(t =>
                `<div class="ok">· <span class="mono">${esc(t.name)}</span> — ${esc(t.description || '')}</div>`).join('');
            el.innerHTML = `<div class="ok">✓ ${esc(data.message)}</div>${list}`;
        } else {
            el.innerHTML = `<div class="err">✕ ${esc(data.message)}</div>`;
        }
    }

    async function saveMcpServer() {
        const cfg = buildMcpConfig();
        if (!cfg.name) { toast('请填写名称', 'error'); return; }
        if (cfg.transport === 'stdio' && !cfg.command) { toast('stdio 需要填写命令', 'error'); return; }
        if (cfg.transport !== 'stdio' && !cfg.url) { toast('请填写 URL', 'error'); return; }
        const resp = await authFetch(`${apiBase()}/mcp/servers`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
        });
        if (!resp) return;
        const data = await resp.json();
        if (data.success) { toast(data.message || '已保存'); closeMcpModal(); loadMcpServers(); }
        else {
            document.getElementById('mcpTestResult').innerHTML = `<div class="err">✕ ${esc(data.message)}</div>`;
            toast(data.message || '保存失败', 'error');
        }
    }

    async function toggleMcpServer(name, enabled) {
        const resp = await authFetch(`${apiBase()}/mcp/servers/${encodeURIComponent(name)}/toggle`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: enabled === true || enabled === 'true' }),
        });
        if (!resp) return;
        if (resp.ok) { toast('已更新'); loadMcpServers(); }
        else { const d = await resp.json().catch(() => ({})); toast(d.detail || '更新失败', 'error'); }
    }

    async function deleteMcpServer(name) {
        if (!confirm(`确定删除 MCP Server '${name}'？`)) return;
        const resp = await authFetch(`${apiBase()}/mcp/servers/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!resp) return;
        if (resp.ok) { toast('已删除'); loadMcpServers(); }
        else { const d = await resp.json().catch(() => ({})); toast(d.detail || '删除失败', 'error'); }
    }

    // ==================== 模态框注入 ====================
    function injectModals() {
        if (document.getElementById('toolModal')) return;

        const toolModal = document.createElement('div');
        toolModal.id = 'toolModal';
        toolModal.className = 'modal-overlay';
        toolModal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h2 class="modal-title" id="toolModalTitle">添加自定义工具</h2>
                    <button class="modal-close" onclick="closeToolModal()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <div class="form-section-title">基本信息</div>
                        <div class="form-group">
                            <label class="form-label">工具文件名 *</label>
                            <input type="text" class="form-input" id="toolId" placeholder="例如: math_pack（仅字母数字下划线连字符）">
                        </div>
                    </div>
                    <div class="form-section">
                        <div class="form-section-title">Python 代码 *</div>
                        <div class="form-group">
                            <textarea class="form-textarea sp-code-textarea" id="toolCode" placeholder="带 docstring 的顶层函数会自动注册为工具。"></textarea>
                        </div>
                        <div class="form-hint">每个不以下划线开头、带 docstring 的顶层函数自动成为一个工具；类型注解用于生成参数。代码在受限子进程沙箱中执行。</div>
                        <div class="sp-validate-box" id="toolValidateResult"></div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeToolModal()">取消</button>
                    <button class="btn btn-secondary" onclick="loadToolTemplate()">加载模板</button>
                    <button class="btn btn-secondary" onclick="validateToolCode()">验证</button>
                    <button class="btn btn-primary" onclick="saveToolCode()">保存</button>
                </div>
            </div>`;
        document.body.appendChild(toolModal);

        const mcpModal = document.createElement('div');
        mcpModal.id = 'mcpModal';
        mcpModal.className = 'modal-overlay';
        mcpModal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h2 class="modal-title" id="mcpModalTitle">添加 MCP Server</h2>
                    <button class="modal-close" onclick="closeMcpModal()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">名称 *</label>
                                <input type="text" class="form-input" id="mcpName" placeholder="my_server（仅字母数字下划线连字符）">
                            </div>
                            <div class="form-group">
                                <label class="form-label">传输方式</label>
                                <select class="form-input" id="mcpTransport" onchange="onMcpTransportChange()">
                                    <option value="stdio">stdio（本地命令）</option>
                                    <option value="streamable_http">streamable_http（远程 URL）</option>
                                    <option value="sse">sse（远程 URL）</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="form-section" id="mcpStdioFields">
                        <div class="form-section-title">本地命令</div>
                        <div class="form-group">
                            <label class="form-label">命令 *</label>
                            <input type="text" class="form-input" id="mcpCommand" placeholder="例如: python 或 npx">
                        </div>
                        <div class="form-group">
                            <label class="form-label">参数（每行一个）</label>
                            <textarea class="form-textarea" id="mcpArgs" placeholder="-m\nyour_server"></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">环境变量（每行 KEY=VALUE）</label>
                            <textarea class="form-textarea" id="mcpEnv" placeholder="API_KEY=xxx"></textarea>
                        </div>
                    </div>
                    <div class="form-section" id="mcpHttpFields" style="display:none;">
                        <div class="form-section-title">远程地址</div>
                        <div class="form-group">
                            <label class="form-label">URL *</label>
                            <input type="text" class="form-input" id="mcpUrl" placeholder="https://host/mcp">
                        </div>
                        <div class="form-group">
                            <label class="form-label">请求头（每行 KEY=VALUE）</label>
                            <textarea class="form-textarea" id="mcpHeaders" placeholder="Authorization=Bearer xxx"></textarea>
                        </div>
                    </div>
                    <div class="sp-validate-box" id="mcpTestResult"></div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeMcpModal()">取消</button>
                    <button class="btn btn-secondary" onclick="testMcpServer()">测试连接</button>
                    <button class="btn btn-primary" onclick="saveMcpServer()">保存</button>
                </div>
            </div>`;
        document.body.appendChild(mcpModal);
    }

    // ==================== 初始化 ====================
    document.addEventListener('DOMContentLoaded', () => {
        bindSubtabs();
        injectModals();
    });

    // 导出全局（供 inline onclick / 头部按钮 / activatePage 调用）
    window.switchSubtab = switchSubtab;
    window.loadCustomTools = loadCustomTools;
    window.openToolModal = openToolModal;
    window.closeToolModal = closeToolModal;
    window.editCustomTool = editCustomTool;
    window.deleteCustomTool = deleteCustomTool;
    window.loadToolTemplate = loadToolTemplate;
    window.validateToolCode = validateToolCode;
    window.saveToolCode = saveToolCode;
    window.loadMcpServers = loadMcpServers;
    window.openMcpModal = openMcpModal;
    window.closeMcpModal = closeMcpModal;
    window.editMcpServer = editMcpServer;
    window.onMcpTransportChange = onMcpTransportChange;
    window.testMcpServer = testMcpServer;
    window.saveMcpServer = saveMcpServer;
    window.toggleMcpServer = toggleMcpServer;
    window.deleteMcpServer = deleteMcpServer;

})();
