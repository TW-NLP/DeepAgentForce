/**
 * DeepAgentForce - ChatGPT-Style Chat JavaScript
 */

// ============ 0. 全局变量 ============
let ws = null;
let isConnected = false;
let isProcessing = false;
let currentThinkingContainer = null;
let currentStreamingAnswer = null;
let currentThinkingSteps = [];  // 当前流式消息的思考步骤数组
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
let currentSessionId = null;
let attachedFiles = [];
let sessionListLoadedViaWebSocket = false; // 🆕 标记会话列表是否已通过 WebSocket 加载
let shouldStickToBottom = true;
let streamingPendingText = '';
let streamingRawText = '';
let streamingRenderFrame = null;
let lastStreamingMarkdownRenderAt = 0;
const STREAMING_MARKDOWN_RENDER_INTERVAL = 140;
let currentProcessingMode = null;
let streamingAnswerPhaseStarted = false;
let currentUploadAbortController = null;

// DOM 元素引用
let messagesWrapper, messagesContainer, welcomeScreen, messageInput, sendButton;
let attachButton, chatFileInput, fileAttachmentsContainer;
let historyList, newChatBtn, sidebarNewChatBtn, statusIndicator, statusText;
let chatTitle, stopButton, regenerateButton;
let lastSubmittedMessage = '';
let lastCompletedUserMessage = '';
let devModeButton, devModeText;
let devModeEnabled = localStorage.getItem('chat_dev_mode') === '1';

// ============ 1. DOM 初始化 ============
function initDOM() {
    messagesWrapper = document.getElementById('messagesWrapper');
    messagesContainer = document.getElementById('messagesContainer');
    welcomeScreen = document.getElementById('welcomeScreen');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    stopButton = document.getElementById('stopButton');
    regenerateButton = document.getElementById('regenerateButton');
    attachButton = document.getElementById('attachButton');
    chatFileInput = document.getElementById('chatFileInput');
    fileAttachmentsContainer = document.getElementById('fileAttachments');
    historyList = document.getElementById('historyList');
    statusIndicator = document.getElementById('statusIndicator');
    statusText = document.getElementById('statusText');
    chatTitle = document.getElementById('chatTitle');
    devModeButton = document.getElementById('headerDevModeBtn');
    devModeText = document.getElementById('headerDevModeText');
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
    const lastConv = historyData.conversations[historyData.conversations.length - 1];
    lastCompletedUserMessage = lastConv?.user_content || '';

    historyData.conversations.forEach(conv => {
        if (conv.user_content) addMessage('user', conv.user_content);
        if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
        if (conv.ai_content) addMessage('assistant', conv.ai_content);
    });
    updateComposerState();
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
        const lastConv = session.conversation[session.conversation.length - 1];
        lastCompletedUserMessage = lastConv?.user_content || '';
        session.conversation.forEach(conv => {
            if (conv.user_content) addMessage('user', conv.user_content);
            if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
            if (conv.ai_content) addMessage('assistant', conv.ai_content);
        });
        updateComposerState();
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
            const lastConv = data.session.conversations[data.session.conversations.length - 1];
            lastCompletedUserMessage = lastConv?.user_content || '';
            data.session.conversations.forEach(conv => {
                if (conv.user_content) addMessage('user', conv.user_content);
                if (conv.thinking_steps?.length > 0) renderThinkingSteps(conv.thinking_steps);
                if (conv.ai_content) addMessage('assistant', conv.ai_content);
            });
            updateComposerState();
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

function sendWsPayload(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        window.showToast?.('连接尚未就绪，请稍后重试', 'error');
        return false;
    }
    ws.send(JSON.stringify(payload));
    return true;
}

function stopGeneration() {
    if (!isProcessing) return;
    if (currentProcessingMode === 'upload') {
        currentUploadAbortController?.abort();
        return;
    }
    sendWsPayload({ action: 'stop', session_id: currentSessionId });
}

function regenerateLastResponse() {
    if (isProcessing || !lastCompletedUserMessage) return;
    sendMessage(lastCompletedUserMessage);
}

function handleWebSocketMessage(payload) {
    console.log('处理消息类型:', payload.type);
    switch (payload.type) {
        case 'assistant_start':
            handleAssistantStart();
            break;
        case 'answer_start':
            handleAnswerStart(payload.data || {});
            break;
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
        case 'assistant_stopped':
            handleStopped(payload.data?.message || '已停止生成');
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
    updateComposerState();
}

function saveDevMode(enabled) {
    devModeEnabled = enabled;
    localStorage.setItem('chat_dev_mode', enabled ? '1' : '0');
    refreshDevModeUI();
    refreshAllDebugPanels();
}

function toggleDevMode() {
    saveDevMode(!devModeEnabled);
    window.showToast?.(devModeEnabled ? '开发模式已开启' : '开发模式已关闭', 'success');
}

function refreshDevModeUI() {
    if (!devModeButton) return;
    devModeButton.classList.toggle('dev-active', devModeEnabled);
    if (devModeText) {
        devModeText.textContent = devModeEnabled ? '开发模式开' : '开发模式关';
    }
}

function updateComposerState() {
    if (sendButton) {
        sendButton.style.display = isProcessing ? 'none' : 'flex';
        sendButton.disabled = !isConnected || isProcessing;
    }

    if (stopButton) {
        const canStop = isProcessing && (currentProcessingMode === 'ws' || currentProcessingMode === 'upload');
        stopButton.style.display = canStop ? 'flex' : 'none';
        stopButton.disabled = !canStop;
    }

    if (regenerateButton) {
        regenerateButton.style.display = !isProcessing && !!lastCompletedUserMessage ? 'flex' : 'none';
        regenerateButton.disabled = !lastCompletedUserMessage || isProcessing;
    }

    if (attachButton) {
        attachButton.disabled = isProcessing;
        attachButton.style.opacity = isProcessing ? '0.5' : '1';
        attachButton.style.pointerEvents = isProcessing ? 'none' : '';
    }
}

// ============ 4. 消息渲染 ============
function hideWelcomeScreen() {
    if (welcomeScreen && welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
    }
}

function isNearBottom(threshold = 120) {
    if (!messagesContainer) return true;
    const distance = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
    return distance <= threshold;
}

function updateScrollStickiness() {
    shouldStickToBottom = isNearBottom();
}

// ============ 工具函数 ============

// 过滤 AI 流式回答中的残留噪音（最小化处理）
function filterAIResponse(content, removeHtmlImages = false) {
    if (!content || typeof content !== 'string') return content;

    return basicCleanAIResponse(content, removeHtmlImages);
}

function basicCleanAIResponse(content, removeHtmlImages = false) {
    if (!content || typeof content !== 'string') return content;

    let cleaned = content;
    cleaned = cleaned.replace(/```[\w]*\n?[\s\S]*?```/g, '');
    cleaned = cleaned.replace(/```[\s\S]*/g, '');
    if (removeHtmlImages) {
        cleaned = cleaned.replace(/<img[^>]*>/gi, '');
    }
    cleaned = cleaned.replace(/\/(?:Users|home|app|tmp)[^\s,'"\]]+/g, '');
    cleaned = cleaned.replace(/\[[^\]]*\/[^\]]*(?:SKILL\.md|\.py)[^\]]*\]/g, '');
    cleaned = cleaned.replace(/\{\s*"datetime"\s*:\s*".*?"\s*,\s*"date"\s*:\s*".*?"\s*,\s*"time"\s*:\s*".*?"[\s\S]*?"utc_offset"\s*:\s*".*?"\s*\}/g, '');
    return cleaned.trim();
}

function createMessageDebugPanel() {
    const panel = document.createElement('div');
    panel.className = `message-debug ${devModeEnabled ? '' : 'is-hidden'}`;
    panel.innerHTML = `
        <div class="message-debug-header">
            <span>Debug View</span>
            <span class="message-debug-meta">raw vs filtered</span>
        </div>
        <div class="message-debug-body">
            <div class="message-debug-col">
                <div class="message-debug-label">Raw</div>
                <pre class="message-debug-pre" data-debug-raw></pre>
            </div>
            <div class="message-debug-col">
                <div class="message-debug-label">Filtered</div>
                <pre class="message-debug-pre" data-debug-filtered></pre>
            </div>
        </div>
    `;
    return panel;
}

function ensureMessageDebugPanel(messageEl) {
    if (!messageEl) return null;
    let panel = messageEl.querySelector('.message-debug');
    if (!panel) {
        panel = createMessageDebugPanel();
        const contentEl = messageEl.querySelector('.message-content');
        contentEl?.appendChild(panel);
    }
    return panel;
}

function updateMessageDebugPanel(messageEl, rawText = '', filteredText = '', meta = '') {
    const panel = ensureMessageDebugPanel(messageEl);
    if (!panel) return;
    panel.classList.toggle('is-hidden', !devModeEnabled);
    const rawEl = panel.querySelector('[data-debug-raw]');
    const filteredEl = panel.querySelector('[data-debug-filtered]');
    const metaEl = panel.querySelector('.message-debug-meta');
    if (rawEl) rawEl.textContent = rawText || '';
    if (filteredEl) filteredEl.textContent = filteredText || '';
    if (metaEl && meta) metaEl.textContent = meta;
    messageEl.dataset.debugRaw = rawText || '';
    messageEl.dataset.debugFiltered = filteredText || '';
    if (meta) messageEl.dataset.debugMeta = meta;
}

function refreshAllDebugPanels() {
    document.querySelectorAll('.message, .message-final, .message-streaming').forEach((messageEl) => {
        const panel = messageEl.querySelector('.message-debug');
        if (panel) {
            panel.classList.toggle('is-hidden', !devModeEnabled);
        }
    });
}

function addMessage(role, content) {
    hideWelcomeScreen();
    const originalContent = content || '';
    let filteredForDebug = originalContent;

    // AI 消息需要过滤原始内容
    if (role === 'assistant') {
        const filteredContent = filterAIResponse(content);
        content = filteredContent || basicCleanAIResponse(content);
        filteredForDebug = content || originalContent;
    }

    const div = document.createElement('div');
    div.className = `message message-${role}`;

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    let messageHtml;

    if (role === 'assistant') {
        // 过滤原始内容
        let filtered = filterAIResponse(content, false);
        if (!filtered) {
            filtered = basicCleanAIResponse(content, false);
        }
        // 完成后解析 markdown
        if (typeof marked !== 'undefined') {
            let parsed = marked.parse(filtered);
            parsed = filterAIResponse(parsed, true); // 移除 HTML img
            if (!parsed) {
                parsed = basicCleanAIResponse(marked.parse(filtered), true);
            }
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
    if (role === 'assistant') {
        updateMessageDebugPanel(div, originalContent, filteredForDebug, 'final_message');
    }
    scrollToBottom(true);
    return div;
}

function buildStreamingMessageShell() {
    const wrapper = document.createElement('div');
    wrapper.className = 'message message-streaming';
    wrapper.innerHTML = `
        <div class="message-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/></svg>
        </div>
        <div class="message-content">
            <div class="streaming-body">
                <div class="streaming-thinking"></div>
                <div class="streaming-status">
                    <span class="typing-indicator"><span></span><span></span><span></span></span>
                    <span class="streaming-status-text">正在思考</span>
                </div>
                <div class="streaming-final is-hidden is-pending"><span class="streaming-cursor">▎</span></div>
            </div>
        </div>
    `;
    messagesWrapper.appendChild(wrapper);
    return wrapper;
}

function ensureStreamingAnswer(statusText = '正在思考') {
    hideWelcomeScreen();

    if (!currentStreamingAnswer) {
        currentStreamingAnswer = buildStreamingMessageShell();
        currentThinkingSteps = [];
        streamingPendingText = '';
        streamingRawText = '';
        streamingAnswerPhaseStarted = false;
        lastStreamingMarkdownRenderAt = 0;
    }

    setStreamingStatus(statusText);
    scrollToBottom(true);
    return currentStreamingAnswer;
}

function setStreamingStatus(text = '正在思考', mode = 'active') {
    if (!currentStreamingAnswer) return;
    const statusEl = currentStreamingAnswer.querySelector('.streaming-status');
    const textEl = currentStreamingAnswer.querySelector('.streaming-status-text');
    if (!statusEl || !textEl) return;

    statusEl.classList.remove('is-hidden', 'is-complete');
    if (mode === 'complete') {
        statusEl.classList.add('is-complete');
    }
    textEl.textContent = text;
}

function hideStreamingStatus() {
    if (!currentStreamingAnswer) return;
    const statusEl = currentStreamingAnswer.querySelector('.streaming-status');
    if (statusEl) {
        statusEl.classList.add('is-hidden');
    }
}

function cancelStreamingRender() {
    if (streamingRenderFrame) {
        cancelAnimationFrame(streamingRenderFrame);
        streamingRenderFrame = null;
    }
}

function markAnswerPhaseStarted() {
    if (!currentStreamingAnswer) return;
    const finalEl = currentStreamingAnswer.querySelector('.streaming-final');
    if (finalEl) {
        finalEl.classList.remove('is-hidden');
    }
    streamingAnswerPhaseStarted = true;
}

function renderStreamingOutput({ forceMarkdown = false } = {}) {
    streamingRenderFrame = null;

    if (!currentStreamingAnswer) return;

    if (streamingPendingText) {
        streamingRawText += streamingPendingText;
        streamingPendingText = '';
    }

    const finalEl = currentStreamingAnswer.querySelector('.streaming-final');
    if (!finalEl) return;

    let filtered = filterAIResponse(streamingRawText, false);
    if (!filtered) {
        filtered = basicCleanAIResponse(streamingRawText, false);
    }
    if (!filtered.trim()) {
        finalEl.classList.add('is-pending');
        finalEl.innerHTML = '<span class="streaming-cursor">▎</span>';
        updateMessageDebugPanel(currentStreamingAnswer, streamingRawText, filtered, 'streaming_pending');
        scrollToBottom();
        return;
    }

    finalEl.classList.remove('is-hidden');
    finalEl.classList.remove('is-pending');

    const now = performance.now();
    const shouldRenderMarkdown = forceMarkdown || (
        typeof marked !== 'undefined' &&
        now - lastStreamingMarkdownRenderAt >= STREAMING_MARKDOWN_RENDER_INTERVAL
    );

    if (shouldRenderMarkdown) {
        let parsed = marked.parse(filtered);
        parsed = filterAIResponse(parsed, true) || basicCleanAIResponse(parsed, true);
        finalEl.innerHTML = parsed + '<span class="streaming-cursor">▎</span>';
        lastStreamingMarkdownRenderAt = now;
    } else {
        finalEl.innerHTML = escapeHtml(filtered).replace(/\n/g, '<br>') + '<span class="streaming-cursor">▎</span>';
    }

    updateMessageDebugPanel(currentStreamingAnswer, streamingRawText, filtered, 'streaming_answer');
    scrollToBottom();
}

function queueStreamingRender() {
    if (streamingRenderFrame) return;
    streamingRenderFrame = requestAnimationFrame(() => renderStreamingOutput());
}

function handleAssistantStart() {
    ensureStreamingAnswer('正在思考');
}

function handleAnswerStart() {
    ensureStreamingAnswer('正在生成答案');
    markAnswerPhaseStarted();
}

function handleTokenUpdate(token) {
    ensureStreamingAnswer('正在回复');
    if (!streamingAnswerPhaseStarted) {
        markAnswerPhaseStarted();
    }
    streamingPendingText += token;
    queueStreamingRender();
}

function handleDone(finalMessage) {
    if (currentStreamingAnswer) {
        cancelStreamingRender();
        if (streamingPendingText) {
            streamingRawText += streamingPendingText;
            streamingPendingText = '';
        }

        currentStreamingAnswer.className = 'message message-final';
        const finalEl = currentStreamingAnswer.querySelector('.streaming-final');
        const thinkingEl = currentStreamingAnswer.querySelector('.streaming-thinking');
        let cleaned = filterAIResponse(finalMessage || streamingRawText || '', false);
        if (!cleaned) {
            cleaned = basicCleanAIResponse(finalMessage || streamingRawText || '', false);
        }

        if (cleaned.trim()) {
            finalEl.classList.remove('is-hidden');
            finalEl.classList.remove('is-pending');
            finalEl.innerHTML = typeof marked !== 'undefined'
                ? (filterAIResponse(marked.parse(cleaned), true) || basicCleanAIResponse(marked.parse(cleaned), true))
                : escapeHtml(cleaned).replace(/\n/g, '<br>');
        } else {
            finalEl.style.display = 'none';
        }
        updateMessageDebugPanel(currentStreamingAnswer, finalMessage || streamingRawText || '', cleaned, 'done');

        if (!thinkingEl.querySelector('.thinking-process') || currentThinkingSteps.length === 0) {
            thinkingEl.style.display = 'none';
        } else {
            // 平滑过渡：先折叠，再淡出，最后隐藏
            const header = thinkingEl.querySelector('.thinking-header');
            const content = thinkingEl.querySelector('.thinking-content');
            if (header && content) {
                header.classList.add('collapsed');
                content.classList.add('collapsed');

                // 延迟添加淡出动画，让折叠动画先完成
                setTimeout(() => {
                    thinkingEl.style.transition = 'opacity 0.4s ease, max-height 0.4s ease';
                    thinkingEl.style.opacity = '0';
                    thinkingEl.style.maxHeight = '0';
                    thinkingEl.style.overflow = 'hidden';

                    setTimeout(() => {
                        thinkingEl.style.display = 'none';
                        thinkingEl.style.opacity = '1';
                        thinkingEl.style.maxHeight = '';
                        thinkingEl.style.transition = '';
                    }, 400);
                }, 300);
            }
        }

        const statusEl = currentStreamingAnswer.querySelector('.streaming-status');
        setStreamingStatus('回复完成', 'complete');
        setTimeout(() => statusEl?.classList.add('is-hidden'), 900);

        currentStreamingAnswer = null;
        currentThinkingSteps = [];
        streamingPendingText = '';
        streamingRawText = '';
        streamingRenderFrame = null;
        streamingAnswerPhaseStarted = false;
    }

    isProcessing = false;
    currentProcessingMode = null;
    currentUploadAbortController = null;
    lastCompletedUserMessage = lastSubmittedMessage || lastCompletedUserMessage;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) { messageInput.disabled = false; messageInput.focus(); }
    updateComposerState();
    loadSavedHistory();
}

function handleStopped(message = '已停止生成') {
    const hadStreamingAnswer = !!currentStreamingAnswer;
    if (currentStreamingAnswer) {
        cancelStreamingRender();
        if (streamingPendingText) {
            streamingRawText += streamingPendingText;
            streamingPendingText = '';
        }

        const finalEl = currentStreamingAnswer.querySelector('.streaming-final');
        const statusEl = currentStreamingAnswer.querySelector('.streaming-status');
        let partial = filterAIResponse(streamingRawText || '', false);
        if (!partial) {
            partial = basicCleanAIResponse(streamingRawText || '', false);
        }

        currentStreamingAnswer.className = 'message message-final';

        if (partial.trim()) {
            finalEl.classList.remove('is-hidden');
            finalEl.classList.remove('is-pending');
            finalEl.innerHTML = typeof marked !== 'undefined'
                ? (filterAIResponse(marked.parse(partial), true) || basicCleanAIResponse(marked.parse(partial), true))
                : escapeHtml(partial).replace(/\n/g, '<br>');
        } else {
            finalEl.innerHTML = `<span style="color: var(--text-tertiary);">${escapeHtml(message)}</span>`;
        }
        updateMessageDebugPanel(currentStreamingAnswer, streamingRawText || '', partial || message, 'stopped');

        if (statusEl) {
            statusEl.classList.remove('is-hidden');
        }
        setStreamingStatus('已停止生成', 'complete');
        setTimeout(() => statusEl?.classList.add('is-hidden'), 1200);

        currentStreamingAnswer = null;
    }
    if (!hadStreamingAnswer) {
        addMessage('assistant', message);
    }

    currentThinkingSteps = [];
    streamingPendingText = '';
    streamingRawText = '';
    streamingRenderFrame = null;
    streamingAnswerPhaseStarted = false;
    isProcessing = false;
    currentProcessingMode = null;
    currentUploadAbortController = null;
    lastCompletedUserMessage = lastSubmittedMessage || lastCompletedUserMessage;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) {
        messageInput.disabled = false;
        messageInput.focus();
    }
    updateComposerState();
}

function handleError(msg) {
    cancelStreamingRender();
    if (currentStreamingAnswer) {
        currentStreamingAnswer.remove();
        currentStreamingAnswer = null;
        currentThinkingSteps = [];
    }
    streamingPendingText = '';
    streamingRawText = '';
    streamingAnswerPhaseStarted = false;
    addMessage('assistant', `❌ 错误: ${escapeHtml(msg)}`);
    isProcessing = false;
    currentProcessingMode = null;
    currentUploadAbortController = null;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
    updateComposerState();
}

// ============ 5. 步骤处理（Claude/Gemini 风格） ============
function handleStepUpdate(payload) {
    hideWelcomeScreen();
    const { step, title, description } = payload.data || payload;
    const timestamp = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    ensureStreamingAnswer(title || '正在思考');

    const thinkingEl = currentStreamingAnswer.querySelector('.streaming-thinking');
    if (!thinkingEl.querySelector('.thinking-process')) {
        thinkingEl.innerHTML = `
            <div class="thinking-process">
                <div class="thinking-header collapsed" onclick="toggleThinking(this)">
                    <span class="thinking-toggle">▼</span>
                    <span class="thinking-title">思考过程</span>
                    <span class="thinking-badge">0 步</span>
                </div>
                <div class="thinking-content collapsed"></div>
            </div>
        `;
    }

    const header = thinkingEl.querySelector('.thinking-header');
    const content = thinkingEl.querySelector('.thinking-content');
    const badge = thinkingEl.querySelector('.thinking-badge');

    const cleanDesc = description ? cleanToolDescription(description) : '';
    setStreamingStatus(title || '正在思考');

    const stepDiv = document.createElement('div');
    stepDiv.className = 'thinking-step is-new';
    stepDiv.innerHTML = `
        <span class="step-icon">${getStepIcon(step)}</span>
        <div class="step-body">
            <div class="step-title">${escapeHtml(title || '处理中')}</div>
            ${cleanDesc ? `<div class="step-desc">${cleanDesc}</div>` : ''}
            <div class="step-time">${timestamp}</div>
        </div>
    `;
    content.appendChild(stepDiv);

    currentThinkingSteps.push({ step, title, description: cleanDesc, timestamp });
    badge.textContent = `${currentThinkingSteps.length} 步`;

    setTimeout(() => stepDiv.classList.remove('is-new'), 1500);

    if (header.classList.contains('collapsed')) {
        header.classList.remove('collapsed');
        content.classList.remove('collapsed');
    }

    scrollToBottom();
}

// 清理工具返回的原始 description（后端已不输出原始内容，此函数保留以兼容接口）
function cleanToolDescription(raw) {
    if (!raw) return '';
    return raw.trim().substring(0, 300);
}

// ============ 6. 思考过程（历史记录用） ============
function renderThinkingSteps(steps) {
    if (!steps?.length) return;

    const container = document.createElement('div');
    container.className = 'thinking-process';
    container.innerHTML = `
        <div class="thinking-header" onclick="toggleThinking(this)">
            <span class="thinking-toggle">▼</span>
            <span class="thinking-title">思考过程</span>
            <span class="thinking-badge">${steps.length} 步</span>
        </div>
        <div class="thinking-content"></div>
    `;

    const content = container.querySelector('.thinking-content');
    steps.forEach(step => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'thinking-step';
        stepDiv.innerHTML = `
            <span class="step-icon">${getStepIcon(step.step_type)}</span>
            <div class="step-body">
                <div class="step-title">${escapeHtml(step.title || '处理中')}</div>
                ${step.description ? `<div class="step-desc">${escapeHtml(step.description)}</div>` : ''}
                ${step.timestamp ? `<div class="step-time">${step.timestamp}</div>` : ''}
            </div>
        `;
        content.appendChild(stepDiv);
    });

    messagesWrapper.appendChild(container);
    scrollToBottom();
}

function toggleThinking(header) {
    const content = header.nextElementSibling;
    const isCollapsed = content.classList.toggle('collapsed');
    header.classList.toggle('collapsed', isCollapsed);
}

function getStepIcon(step) {
    if (!step) return '⚙️';
    const s = step.toLowerCase();
    if (s.includes('init') || s.includes('开始')) return '🚀';
    if (s.includes('summarize') || s.includes('生成答案')) return '✍️';
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

async function readSseResponse(response, handlers = {}) {
    if (!response.ok) {
        let detail = '';
        try {
            detail = await response.text();
        } catch (_) {
            detail = '';
        }
        throw new Error(detail || `请求失败 (${response.status})`);
    }

    if (!response.body) {
        throw new Error('服务器未返回流式内容');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const dispatchEvent = async (eventName, payload) => {
        const handler = handlers[eventName] || handlers.default;
        if (handler) {
            return handler(payload, eventName);
        }
    };

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let separatorIndex = buffer.indexOf('\n\n');
        while (separatorIndex !== -1) {
            const rawEvent = buffer.slice(0, separatorIndex);
            buffer = buffer.slice(separatorIndex + 2);

            let eventName = 'message';
            const dataLines = [];
            for (const line of rawEvent.split(/\r?\n/)) {
                if (line.startsWith('event:')) {
                    eventName = line.slice(6).trim() || 'message';
                } else if (line.startsWith('data:')) {
                    dataLines.push(line.slice(5).trimStart());
                }
            }

            if (dataLines.length > 0) {
                let payload = {};
                const dataText = dataLines.join('\n');
                try {
                    payload = JSON.parse(dataText);
                } catch (_) {
                    payload = { content: dataText };
                }
                await dispatchEvent(eventName, payload);
            }

            separatorIndex = buffer.indexOf('\n\n');
        }
    }

    if (buffer.trim()) {
        let eventName = 'message';
        const dataLines = [];
        for (const line of buffer.split(/\r?\n/)) {
            if (line.startsWith('event:')) {
                eventName = line.slice(6).trim() || 'message';
            } else if (line.startsWith('data:')) {
                dataLines.push(line.slice(5).trimStart());
            }
        }
        if (dataLines.length > 0) {
            let payload = {};
            const dataText = dataLines.join('\n');
            try {
                payload = JSON.parse(dataText);
            } catch (_) {
                payload = { content: dataText };
            }
            await dispatchEvent(eventName, payload);
        }
    }
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

    lastSubmittedMessage = message;

    if (attachedFiles.length > 0) {
        // 带附件上传
        try {
            const formData = new FormData();
            formData.append('message', message);
            if (currentSessionId) formData.append('session_id', currentSessionId);
            attachedFiles.forEach(file => formData.append('files', file));

            addMessage('user', message + (attachedFiles.length ? `\n\n📎 ${attachedFiles.map(f => f.name).join(', ')}` : ''));
            ensureStreamingAnswer('正在处理附件');

            isProcessing = true;
            currentProcessingMode = 'upload';
            currentUploadAbortController = new AbortController();
            messageInput.disabled = true;
            updateComposerState();

            const token = localStorage.getItem('access_token');
            const response = await fetch(`${getApiUrl()}/chat/upload/stream`, {
                method: 'POST',
                headers: token ? { 'Authorization': `Bearer ${token}` } : {},
                body: formData,
                signal: currentUploadAbortController.signal
            });

            await readSseResponse(response, {
                start: (data) => {
                    if (data?.session_id) {
                        currentSessionId = data.session_id;
                    }
                },
                step: (data) => {
                    handleStepUpdate({ data });
                },
                answer_start: (data) => {
                    handleAnswerStart(data);
                },
                token: (data) => {
                    handleTokenUpdate(data?.content || '');
                },
                done: (data) => {
                    attachedFiles = [];
                    renderFileAttachments();
                    if (data?.session_id) {
                        currentSessionId = data.session_id;
                    }
                    handleDone(data?.message || streamingRawText || '');
                },
                assistant_stopped: (data) => {
                    handleStopped(data?.message || '已停止生成');
                },
                error: (data) => {
                    handleError(data?.message || data?.error || '文件上传失败');
                }
            });
        } catch (error) {
            if (error?.name === 'AbortError') {
                handleStopped('已停止生成');
                return;
            }
            handleError('文件上传失败: ' + error.message);
        } finally {
            currentUploadAbortController = null;
        }
    } else {
        // 普通消息
        addMessage('user', message);
        ensureStreamingAnswer('正在思考');

        const payload = { action: 'message', message };
        if (currentSessionId) payload.session_id = currentSessionId;
        if (!sendWsPayload(payload)) {
            return;
        }

        isProcessing = true;
        currentProcessingMode = 'ws';
        messageInput.disabled = true;
        updateComposerState();
    }

    if (!text) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }

    shouldStickToBottom = true;
}

function scrollToBottom(force = false) {
    requestAnimationFrame(() => {
        if (messagesContainer && (force || shouldStickToBottom)) {
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
    cancelStreamingRender();
    currentStreamingAnswer = null;
    currentThinkingSteps = [];
    lastSubmittedMessage = '';
    streamingPendingText = '';
    streamingRawText = '';
    streamingAnswerPhaseStarted = false;
    isProcessing = false;
    currentProcessingMode = null;
    currentUploadAbortController = null;
    attachedFiles = [];
    renderFileAttachments();
    updateComposerState();
}

function startNewChat() {
    currentSessionId = null;
    currentThinkingSteps = [];
    lastSubmittedMessage = '';
    lastCompletedUserMessage = '';
    if (chatTitle) chatTitle.textContent = '新对话';
    messagesWrapper.innerHTML = '';
    messagesWrapper.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';
    cancelStreamingRender();
    currentStreamingAnswer = null;
    streamingPendingText = '';
    streamingRawText = '';
    streamingAnswerPhaseStarted = false;
    isProcessing = false;
    currentProcessingMode = null;
    currentUploadAbortController = null;
    attachedFiles = [];
    renderFileAttachments();
    if (messageInput) {
        messageInput.value = '';
        messageInput.focus();
    }
    updateComposerState();
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

    if (devModeButton) {
        devModeButton.addEventListener('click', toggleDevMode);
    }

    if (stopButton) {
        stopButton.addEventListener('click', stopGeneration);
    }

    if (regenerateButton) {
        regenerateButton.addEventListener('click', regenerateLastResponse);
    }

    if (messageInput) {
        messageInput.addEventListener('input', autoResizeTextarea);
        messageInput.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
                e.preventDefault();
                toggleDevMode();
                return;
            }
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
    if (messagesContainer) {
        messagesContainer.addEventListener('scroll', updateScrollStickiness, { passive: true });
        shouldStickToBottom = true;
    }
    updateComposerState();
    refreshDevModeUI();
    connectWebSocket();
    loadSavedHistory();

    // 自动聚焦输入框
    if (messageInput) messageInput.focus();

    console.log('✅ DeepAgentForce Chat 已加载');
});
