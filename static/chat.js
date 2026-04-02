/**
 * DeepAgentForce - ChatGPT-Style Chat JavaScript
 */

// ============ 0. 全局变量 ============
let ws = null;
let isConnected = false;
let isProcessing = false;
let currentThinkingContainer = null;
let currentStreamingAnswer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
let currentSessionId = null;
let attachedFiles = [];
let sessionListLoadedViaWebSocket = false; // 🆕 标记会话列表是否已通过 WebSocket 加载

// DOM 元素引用
let messagesWrapper, messagesContainer, welcomeScreen, messageInput, sendButton;
let attachButton, chatFileInput, fileAttachmentsContainer;
let historyList, newChatBtn, sidebarNewChatBtn, statusIndicator, statusText;
let chatTitle;

// ============ 1. DOM 初始化 ============
function initDOM() {
    messagesWrapper = document.getElementById('messagesWrapper');
    messagesContainer = document.getElementById('messagesContainer');
    welcomeScreen = document.getElementById('welcomeScreen');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    attachButton = document.getElementById('attachButton');
    chatFileInput = document.getElementById('chatFileInput');
    fileAttachmentsContainer = document.getElementById('fileAttachments');
    historyList = document.getElementById('historyList');
    statusIndicator = document.getElementById('statusIndicator');
    statusText = document.getElementById('statusText');
    chatTitle = document.getElementById('chatTitle');
}

// ============ 2. 历史记录 ============
async function loadSavedHistory() {
    // 🆕 如果已经通过 WebSocket 加载了会话列表，不再通过 API 重复加载
    if (sessionListLoadedViaWebSocket) {
        return;
    }

    try {
        const response = await window.auth?.authFetch?.(`${getApiUrl()}/history/saved`) || 
                         fetch(`${getApiUrl()}/history/saved`, {
                             headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
                         });

        if (!response.ok) return;

        const data = await response.json();

        if (historyList) {
            historyList.innerHTML = '';
        }

        if (data.success && Array.isArray(data.sessions) && data.sessions.length > 0) {
            const sortedSessions = [...data.sessions].sort((a, b) =>
                new Date(b.updated_at) - new Date(a.updated_at)
            );

            sortedSessions.forEach((session) => {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.dataset.sessionId = session.session_id;

                if (currentSessionId === session.session_id) {
                    li.classList.add('active');
                }

                let title = session.title || '新对话';
                if ((title === '新对话' || title === '历史对话') && session.conversation?.length > 0) {
                    const firstMsg = session.conversation[0].user_content;
                    if (firstMsg) {
                        title = firstMsg.length > 30 ? firstMsg.substring(0, 30) + '...' : firstMsg;
                    }
                }

                const contentDiv = document.createElement('div');
                contentDiv.className = 'history-item-content';
                contentDiv.innerHTML = `<span class="history-item-title">${escapeHtml(title)}</span>`;

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'history-delete-btn';
                deleteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (confirm(`确定要删除会话"${title}"吗？`)) {
                        deleteSession(session.session_id);
                    }
                };
                contentDiv.appendChild(deleteBtn);
                li.appendChild(contentDiv);

                li.onclick = () => restoreSession(session);
                historyList.appendChild(li);
            });
        } else {
            historyList.innerHTML = '<li class="history-empty">暂无历史记录</li>';
        }
    } catch (error) {
        console.error("加载历史记录失败:", error);
    }
}

// 处理从 WebSocket 收到的历史记录（页面加载时）
function renderSessionHistory(historyData) {
    if (!historyData?.conversations?.length) return;

    hideWelcomeScreen();

    historyData.conversations.forEach(conv => {
        if (conv.user_content) addMessage('user', conv.user_content);
        if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
        if (conv.ai_content) addMessage('assistant', conv.ai_content);
    });
}

// 处理 WebSocket 发送的会话列表（用于侧边栏显示）
function renderSessionList(sessions) {
    if (!sessions || !Array.isArray(sessions)) return;

    // 🆕 标记会话列表已通过 WebSocket 加载
    sessionListLoadedViaWebSocket = true;

    if (historyList) {
        historyList.innerHTML = '';
    }

    if (sessions.length === 0) {
        historyList.innerHTML = '<li class="history-empty">暂无历史记录</li>';
        return;
    }

    const sortedSessions = [...sessions].sort((a, b) =>
        new Date(b.updated_at || 0) - new Date(a.updated_at || 0)
    );

    sortedSessions.forEach((session) => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.dataset.sessionId = session.session_id;

        if (currentSessionId === session.session_id) {
            li.classList.add('active');
        }

        let title = session.title || '新对话';
        if ((title === '新对话' || title === '历史对话') && session.conversation?.length > 0) {
            const firstMsg = session.conversation[0].user_content;
            if (firstMsg) {
                title = firstMsg.length > 30 ? firstMsg.substring(0, 30) + '...' : firstMsg;
            }
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'history-item-content';
        contentDiv.innerHTML = `<span class="history-item-title">${escapeHtml(title)}</span>`;

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'history-delete-btn';
        deleteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            if (confirm(`确定要删除会话"${title}"吗？`)) {
                deleteSession(session.session_id);
            }
        };
        contentDiv.appendChild(deleteBtn);
        li.appendChild(contentDiv);

        li.onclick = () => restoreSession(session);
        historyList.appendChild(li);
    });
}

async function deleteSession(sessionId) {
    try {
        const response = await window.auth?.authFetch?.(`${getApiUrl()}/history/session/${sessionId}`, {
            method: 'DELETE'
        }) || fetch(`${getApiUrl()}/history/session/${sessionId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
        });

        if (response.ok) {
            if (currentSessionId === sessionId) {
                currentSessionId = null;
                if (chatTitle) chatTitle.textContent = '新对话';
                messagesWrapper.innerHTML = '';
                messagesWrapper.appendChild(welcomeScreen);
                welcomeScreen.style.display = 'flex';
            }
            loadSavedHistory();
            if (window.showToast) window.showToast('会话已删除', 'success');
        }
    } catch (error) {
        console.error('删除会话失败:', error);
    }
}

function restoreSession(session) {
    currentSessionId = session.session_id;
    if (chatTitle) chatTitle.textContent = session.title || '对话';

    // 更新高亮
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.toggle('active', item.dataset.sessionId === currentSessionId);
    });

    resetChatUI();

    // 如果本地有对话数据，直接渲染
    if (session.conversation?.length > 0) {
        session.conversation.forEach(conv => {
            if (conv.user_content) addMessage('user', conv.user_content);
            if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
            if (conv.ai_content) addMessage('assistant', conv.ai_content);
        });
    } else {
        // 本地没有数据，通过 API 加载完整会话
        loadSessionMessages(session.session_id);
    }
}

// 从 API 加载会话消息
async function loadSessionMessages(sessionId) {
    try {
        const response = await window.auth?.authFetch?.(`${getApiUrl()}/history/session/${sessionId}`) ||
                         fetch(`${getApiUrl()}/history/session/${sessionId}`, {
                             headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
                         });

        if (!response.ok) return;

        const data = await response.json();
        if (data.success && data.session?.conversations?.length > 0) {
            data.session.conversations.forEach(conv => {
                if (conv.user_content) addMessage('user', conv.user_content);
                if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
                if (conv.ai_content) addMessage('assistant', conv.ai_content);
            });
        }
    } catch (error) {
        console.error('加载会话消息失败:', error);
    }
}

// ============ 3. WebSocket ============
function getWsUrl() {
    if (CONFIG?._wsBase) {
        const token = localStorage.getItem('access_token');
        const sep = CONFIG._wsBase.includes('?') ? '&' : '?';
        return token ? `${CONFIG._wsBase}${sep}token=${encodeURIComponent(token)}` : CONFIG._wsBase;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const token = localStorage.getItem('access_token');
    const url = `${protocol}//${host}:8000/ws/stream`;
    return token ? `${url}?token=${encodeURIComponent(token)}` : url;
}

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

    // 🆕 重连时重置会话列表加载标志，确保能接收新的会话列表
    sessionListLoadedViaWebSocket = false;

    const wsUrl = getWsUrl();
    console.log('WebSocket 连接 URL:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isConnected = true;
        reconnectAttempts = 0;
        updateStatus(true);
        console.log('WebSocket 已连接');
    };

    ws.onmessage = (event) => {
        console.log('WebSocket 收到消息:', event.data);
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = () => updateStatus(false);

    ws.onclose = () => {
        isConnected = false;
        updateStatus(false);
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            setTimeout(connectWebSocket, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
        }
    };
}

function handleWebSocketMessage(payload) {
    console.log('处理消息类型:', payload.type);
    switch (payload.type) {
        case 'step':
            handleStepUpdate(payload);
            break;
        case 'token':
            handleTokenUpdate(payload.content || payload.data?.content || '');
            break;
        case 'done':
            if (payload.data?.session_id) currentSessionId = payload.data.session_id;
            handleDone(payload.data?.message || payload.message);
            break;
        case 'error':
            handleError(payload.data?.message || payload.message);
            break;
        case 'history':
            // 处理页面加载时从后端发送的历史记录
            if (payload.data?.conversations?.length > 0) {
                renderSessionHistory(payload.data);
            }
            break;
        case 'session_list':
            // 处理 WebSocket 发送的会话列表（用于侧边栏显示）
            console.log('收到 session_list 消息:', payload.data);
            if (payload.data?.sessions?.length > 0) {
                renderSessionList(payload.data.sessions);
            }
            break;
    }
}

function updateStatus(connected) {
    if (statusIndicator) {
        statusIndicator.className = `status-indicator ${connected ? 'connected' : ''}`;
        if (statusText) statusText.textContent = connected ? '在线' : '连接中';
    }
}

// ============ 4. 消息渲染 ============
function hideWelcomeScreen() {
    if (welcomeScreen && welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
    }
}

// ============ 工具函数 ============

// 过滤 AI 响应的原始内部内容，只保留友好的最终结果
function filterAIResponse(content, removeHtmlImages = false) {
    if (!content || typeof content !== 'string') return content;

    // 移除 Skill 定义块 (--- ... --- 格式)
    content = content.replace(/^---[\s\S]*?---\s*/gm, '');

    // 移除命令行格式的内容
    content = content.replace(/^(\$\s*|```bash\n|```sh\n|```shell\n)/gm, '');
    content = content.replace(/(\$\s*python.*$)/gm, '');
    content = content.replace(/^(python.*$)/gm, '');

    // 移除 JSON 格式的工具调用定义
    content = content.replace(/^(\{[\s\S]*?"name":\s*"[^"]+")/gm, '');
    content = content.replace(/(```json\n[\s\S]*?```)/g, '');

    // 移除代码块中的命令
    content = content.replace(/(python \/.*?\.py)/g, '');

    // 移除 Markdown 图片语法
    content = content.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');

    // 移除搜索日志信息（多种格式）
    content = content.replace(/成功从\s*[\/].*?加载配置\s*/g, '');
    content = content.replace(/🔍\s*Searching for:.*$/gm, '');
    content = content.replace(/Searching for:.*$/gm, '');

    // 移除 JSON 代码块（多行）
    content = content.replace(/```json\s*[\s\S]*?```/g, '');

    // 移除 "根据搜索结果" 之后的所有内容
    content = content.replace(/根据搜索结果[\s\S]*$/gm, '');
    content = content.replace(/根据查询结果[\s\S]*$/gm, '');

    // 使用更强大的 JSON 移除逻辑：检测包含 query/total_results/results 的行，然后移除连续的多行 JSON
    const lines = content.split('\n');
    const filteredLines = [];
    let inJsonBlock = false;
    let braceCount = 0;
    let skipUntilMeaningfulContent = false;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // 检测是否是 JSON 开始的行
        if (!inJsonBlock && (line.includes('"query"') || line.includes('"total_results"') || line.includes('"results"'))) {
            inJsonBlock = true;
            braceCount = 0;
            skipUntilMeaningfulContent = false;
        }

        if (inJsonBlock) {
            // 统计大括号
            braceCount += (line.match(/\{/g) || []).length;
            braceCount -= (line.match(/\}/g) || []).length;

            // 如果 JSON 结束
            if (braceCount <= 0 && line.includes('}')) {
                inJsonBlock = false;
            }
            continue; // 跳过这行
        }

        // 检测 "根据搜索结果" 之后的内容
        if (line.includes('根据搜索结果') || line.includes('根据查询结果')) {
            skipUntilMeaningfulContent = true;
            continue;
        }

        // 如果正在跳过，检查是否有有意义的内容（新段落）
        if (skipUntilMeaningfulContent) {
            // 如果是空行或只有空白，继续跳过
            if (line.trim() === '' || line.match(/^\s+$/)) {
                continue;
            }
            // 如果是正常内容段落（不以 { 或 [ 开头）
            if (!line.trim().startsWith('{') && !line.trim().startsWith('[')) {
                skipUntilMeaningfulContent = false;
            } else {
                continue;
            }
        }

        filteredLines.push(line);
    }

    content = filteredLines.join('\n');

    // 如果需要，移除 HTML img 标签
    if (removeHtmlImages) {
        content = content.replace(/<img[^>]*>/gi, '');
    }

    // 清理多余的空行
    content = content.replace(/\n{3,}/g, '\n\n');

    // 去除首尾空白
    content = content.trim();

    return content;
}

function addMessage(role, content) {
    hideWelcomeScreen();

    // AI 消息需要过滤原始内容
    if (role === 'assistant') {
        content = filterAIResponse(content);
    }

    const div = document.createElement('div');
    div.className = `message message-${role}`;

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    let messageHtml;

    if (role === 'assistant') {
        // 过滤原始内容
        let filtered = filterAIResponse(content, false);
        // 完成后解析 markdown
        if (typeof marked !== 'undefined') {
            let parsed = marked.parse(filtered);
            parsed = filterAIResponse(parsed, true); // 移除 HTML img
            messageHtml = parsed;
        } else {
            messageHtml = escapeHtml(filtered);
        }
    } else {
        messageHtml = escapeHtml(content);
    }

    div.innerHTML = `
        <div class="message-avatar">
            ${role === 'user'
                ? `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
                : `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`
            }
        </div>
        <div class="message-content">
            <div class="message-text">${messageHtml}</div>
        </div>
    `;

    messagesWrapper.appendChild(div);
    scrollToBottom();
    return div;
}

function handleTokenUpdate(token) {
    const statusEl = document.getElementById('processingStatus');
    if (statusEl) statusEl.remove();

    if (!currentStreamingAnswer) {
        currentStreamingAnswer = document.createElement('div');
        currentStreamingAnswer.className = 'message message-assistant';
        const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        currentStreamingAnswer.innerHTML = `
            <div class="message-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <div class="message-content">
                <div class="message-text streaming" data-raw=""><span class="cursor-blink">▎</span></div>
            </div>
        `;
        messagesWrapper.appendChild(currentStreamingAnswer);
    }

    const contentDiv = currentStreamingAnswer.querySelector('.message-content .message-text');
    const cursor = contentDiv.querySelector('.cursor-blink');
    const currentRaw = contentDiv.dataset.raw || '';
    const newRaw = currentRaw + token;
    contentDiv.dataset.raw = newRaw;

    if (cursor) cursor.remove();

    // 流式输出时先过滤不需要的内容，只做 HTML 转义
    let filteredRaw = filterAIResponse(newRaw, false);
    filteredRaw = escapeHtml(filteredRaw).replace(/\n/g, '<br>');
    contentDiv.innerHTML = filteredRaw + '<span class="cursor-blink">▎</span>';

    scrollToBottom();
}

function handleDone(finalMessage) {
    const statusEl = document.getElementById('processingStatus');
    if (statusEl) statusEl.remove();

    // 过滤原始内容
    finalMessage = filterAIResponse(finalMessage);

    if (currentStreamingAnswer) {
        const contentDiv = currentStreamingAnswer.querySelector('.message-content .message-text');
        contentDiv.classList.remove('streaming');
        if (finalMessage) {
            // 完成后过滤并解析 markdown
            let filtered = filterAIResponse(finalMessage, false);
            if (typeof marked !== 'undefined') {
                let parsed = marked.parse(filtered);
                parsed = filterAIResponse(parsed, true); // 移除 HTML img
                contentDiv.innerHTML = parsed;
            } else {
                contentDiv.innerHTML = escapeHtml(filtered);
            }
        }
    } else if (finalMessage) {
        addMessage('assistant', finalMessage);
    }

    currentStreamingAnswer = null;
    isProcessing = false;

    if (sendButton) sendButton.disabled = false;
    if (messageInput) {
        messageInput.disabled = false;
        messageInput.focus();
    }

    loadSavedHistory();
}

function handleError(msg) {
    addMessage('assistant', `❌ 错误: ${escapeHtml(msg)}`);
    isProcessing = false;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
}

// ============ 5. 步骤处理 ============
function handleStepUpdate(payload) {
    hideWelcomeScreen();
    const { step, title, description } = payload.data || payload;

    if (currentStreamingAnswer) {
        const textEl = currentStreamingAnswer.querySelector('.processing-text');
        if (textEl) textEl.textContent = title || '处理中';
    } else {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'message message-assistant';
        statusDiv.id = 'processingStatus';
        statusDiv.innerHTML = `
            <div class="message-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <div class="message-content">
                <div class="processing-indicator">
                    <div class="processing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
                    <span class="processing-text">${escapeHtml(title || '处理中')}</span>
                </div>
                ${description ? `<div class="processing-description">${escapeHtml(description)}</div>` : ''}
            </div>
        `;
        messagesWrapper.appendChild(statusDiv);
        scrollToBottom();
    }
}

// ============ 6. 思考过程 ============
function renderThinkingSteps(steps) {
    if (!steps?.length) return;

    const container = document.createElement('div');
    container.className = 'thinking-process';
    container.innerHTML = `
        <div class="thinking-header" onclick="toggleThinking(this)">
            <span class="thinking-toggle">▼</span>
            <span class="thinking-title">思考过程</span>
        </div>
        <div class="thinking-content"></div>
    `;

    const content = container.querySelector('.thinking-content');
    steps.forEach(step => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'thinking-step';
        stepDiv.innerHTML = `
            <span class="step-icon">${getStepIcon(step.step_type)}</span>
            <div class="step-content">
                <div class="step-title">${escapeHtml(step.title || '处理中')}</div>
                ${step.description ? `<div class="step-description">${escapeHtml(step.description)}</div>` : ''}
            </div>
        `;
        content.appendChild(stepDiv);
    });

    messagesWrapper.appendChild(container);
    scrollToBottom();
}

function toggleThinking(header) {
    const content = header.nextElementSibling;
    const toggle = header.querySelector('.thinking-toggle');
    const isCollapsed = content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed', isCollapsed);
    toggle.textContent = isCollapsed ? '▶' : '▼';
}

function getStepIcon(step) {
    if (!step) return '⚙️';
    const s = step.toLowerCase();
    if (s.includes('init') || s.includes('开始')) return '🚀';
    if (s.includes('tool') && s.includes('start')) return '🔧';
    if (s.includes('tool') && s.includes('end')) return '✅';
    if (s.includes('finish') || s.includes('结束')) return '🎯';
    if (s.includes('error')) return '❌';
    return '⚙️';
}

// ============ 7. 文件上传 ============
function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        pdf: '📄', doc: '📝', docx: '📝', txt: '📃', md: '📋', csv: '📊'
    };
    return icons[ext] || '📎';
}

function renderFileAttachments() {
    if (!fileAttachmentsContainer) return;

    if (attachedFiles.length === 0) {
        fileAttachmentsContainer.innerHTML = '';
        fileAttachmentsContainer.style.display = 'none';
        return;
    }

    fileAttachmentsContainer.style.display = 'flex';
    fileAttachmentsContainer.innerHTML = attachedFiles.map((file, index) => `
        <div class="file-attachment">
            <span class="file-icon">${getFileIcon(file.name)}</span>
            <span class="file-name" title="${file.name}">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <span class="file-remove" onclick="removeAttachment(${index})">✕</span>
        </div>
    `).join('');
}

function removeAttachment(index) {
    attachedFiles.splice(index, 1);
    renderFileAttachments();
}

// ============ 8. 发送消息 ============
async function sendMessage(text = null) {
    const message = text || messageInput.value.trim();

    if ((!message && attachedFiles.length === 0) || !isConnected || isProcessing) {
        if (!isConnected && window.showToast) window.showToast("未连接到服务器", "error");
        return;
    }

    // 更新标题
    if (chatTitle && !currentSessionId && message.length > 0) {
        chatTitle.textContent = message.length > 20 ? message.substring(0, 20) + '...' : message;
    }

    if (attachedFiles.length > 0) {
        // 带附件上传
        try {
            const formData = new FormData();
            formData.append('message', message);
            if (currentSessionId) formData.append('session_id', currentSessionId);
            attachedFiles.forEach(file => formData.append('files', file));

            addMessage('user', message + (attachedFiles.length ? `\n\n📎 ${attachedFiles.map(f => f.name).join(', ')}` : ''));

            isProcessing = true;
            sendButton.disabled = true;
            messageInput.disabled = true;

            const token = localStorage.getItem('access_token');
            const response = await fetch(`${getApiUrl()}/chat/upload`, {
                method: 'POST',
                headers: token ? { 'Authorization': `Bearer ${token}` } : {},
                body: formData
            });

            const result = await response.json();
            attachedFiles = [];
            renderFileAttachments();

            if (result.session_id) currentSessionId = result.session_id;
            loadSavedHistory();

            if (result.message) {
                handleDone(result.message);
            } else {
                isProcessing = false;
                sendButton.disabled = false;
                messageInput.disabled = false;
            }
        } catch (error) {
            handleError('文件上传失败: ' + error.message);
        }
    } else {
        // 普通消息
        addMessage('user', message);

        const payload = { message };
        if (currentSessionId) payload.session_id = currentSessionId;
        ws.send(JSON.stringify(payload));

        isProcessing = true;
        sendButton.disabled = true;
        messageInput.disabled = true;
    }

    if (!text) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    });
}

function autoResizeTextarea() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
}

function resetChatUI() {
    messagesWrapper.innerHTML = '';
    currentStreamingAnswer = null;
    isProcessing = false;
    attachedFiles = [];
    renderFileAttachments();
}

function startNewChat() {
    currentSessionId = null;
    if (chatTitle) chatTitle.textContent = '新对话';
    messagesWrapper.innerHTML = '';
    messagesWrapper.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';
    currentStreamingAnswer = null;
    isProcessing = false;
    attachedFiles = [];
    renderFileAttachments();
    if (messageInput) {
        messageInput.value = '';
        messageInput.focus();
    }
    loadSavedHistory();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============ 9. 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    // 检查登录状态
    if (!localStorage.getItem('access_token')) {
        window.location.href = '/login.html';
        return;
    }

    initDOM();

    if (sendButton) {
        sendButton.addEventListener('click', () => sendMessage());
    }

    if (messageInput) {
        messageInput.addEventListener('input', autoResizeTextarea);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (attachButton) {
        attachButton.addEventListener('click', () => chatFileInput?.click());
    }

    if (chatFileInput) {
        chatFileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            const maxFiles = 10;
            const maxSize = 10 * 1024 * 1024;

            for (const file of files) {
                if (attachedFiles.length >= maxFiles) {
                    window.showToast?.('最多只能上传10个文件', 'error');
                    break;
                }
                if (file.size > maxSize) {
                    window.showToast?.(`文件 ${file.name} 超过10MB限制`, 'error');
                    continue;
                }
                attachedFiles.push(file);
            }

            renderFileAttachments();
            chatFileInput.value = '';
        });
    }

    // 快捷提示点击
    document.querySelectorAll('.quick-prompt-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) sendMessage(prompt);
        });
    });

    // 全局函数
    window.toggleThinking = toggleThinking;
    window.removeAttachment = removeAttachment;
    window.startNewChat = startNewChat;

    // 启动
    connectWebSocket();
    loadSavedHistory();

    // 自动聚焦输入框
    if (messageInput) messageInput.focus();

    console.log('✅ DeepAgentForce Chat 已加载');
});
