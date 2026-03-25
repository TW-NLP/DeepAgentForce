/**
 * 对话功能 JavaScript
 */

// ============ 0. 全局变量定义 ============
let ws = null;
let isConnected = false;
let isProcessing = false;
let currentThinkingContainer = null;
let currentStreamingAnswer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// 核心变量：全局 session ID
let currentSessionId = null;

// 文件上传相关变量
let attachedFiles = [];

// DOM 元素引用
let messagesWrapper, messagesArea, welcomeScreen, messageInput, sendButton;
let attachButton, chatFileInput, fileAttachmentsContainer;
let historyList, newChatBtn, sidebarNewChatBtn, statusIndicator, statusText;

// ============ 1. DOM 初始化 ============
function initDOM() {
    messagesWrapper = document.getElementById('messagesWrapper');
    messagesArea = document.getElementById('messagesArea');
    welcomeScreen = document.getElementById('welcomeScreen');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    attachButton = document.getElementById('attachButton');
    chatFileInput = document.getElementById('chatFileInput');
    fileAttachmentsContainer = document.getElementById('fileAttachments');
    historyList = document.getElementById('historyList');
    newChatBtn = document.getElementById('newChatBtn');
    sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn');
    statusIndicator = document.getElementById('statusIndicator');
    statusText = document.getElementById('statusText');
}

// ============ 2. 历史记录加载与管理 ============
async function loadSavedHistory() {
    try {
        console.log("正在加载历史记录...");
        const response = await fetch(`${getApiUrl()}/history/saved`);

        if (!response.ok) {
            console.warn("无法连接到历史记录接口");
            return;
        }

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
                li.dataset.sessionId = session.session_id;  // 🆕 保存 session_id 用于查找

                if (currentSessionId === session.session_id) {
                    li.classList.add('active');
                }

                // 🆕 优先使用后端返回的 title（已根据第一条用户消息生成）
                let title = session.title || '新对话';

                // 兼容旧数据：如果 title 仍是默认值，尝试从对话内容生成
                if ((title === '新对话' || title === '历史对话') && session.conversation && session.conversation.length > 0) {
                    const firstMsg = session.conversation[0].user_content;
                    if (firstMsg) {
                        title = firstMsg.length > 30
                            ? firstMsg.substring(0, 30) + '...'
                            : firstMsg;
                    }
                }

                // 创建标题容器
                const titleContainer = document.createElement('div');
                titleContainer.className = 'history-item-content';
                titleContainer.style.display = 'flex';
                titleContainer.style.alignItems = 'center';
                titleContainer.style.justifyContent = 'space-between';
                titleContainer.style.width = '100%';
                titleContainer.style.gap = '8px';

                // 标题文本
                const titleSpan = document.createElement('span');
                titleSpan.className = 'history-item-title';
                titleSpan.textContent = title;
                titleSpan.style.flex = '1';
                titleSpan.style.overflow = 'hidden';
                titleSpan.style.textOverflow = 'ellipsis';
                titleSpan.style.whiteSpace = 'nowrap';
                titleContainer.appendChild(titleSpan);

                // 🆕 删除按钮
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'history-delete-btn';
                deleteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>`;
                deleteBtn.title = '删除会话';
                deleteBtn.style.display = 'none';
                deleteBtn.style.background = 'none';
                deleteBtn.style.border = 'none';
                deleteBtn.style.cursor = 'pointer';
                deleteBtn.style.color = 'inherit';
                deleteBtn.style.padding = '2px';
                deleteBtn.style.borderRadius = '4px';
                deleteBtn.style.transition = 'all 0.2s';

                // 🆕 删除按钮悬停效果
                deleteBtn.onmouseenter = () => {
                    deleteBtn.style.background = 'rgba(255, 255, 255, 0.2)';
                    deleteBtn.style.color = '#ff4d4f';
                };
                deleteBtn.onmouseleave = () => {
                    deleteBtn.style.background = 'none';
                    deleteBtn.style.color = 'inherit';
                };

                // 🆕 点击删除按钮
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    confirmDeleteSession(session.session_id, title);
                };

                titleContainer.appendChild(deleteBtn);
                li.appendChild(titleContainer);

                // 🆕 悬停时显示删除按钮
                li.onmouseenter = () => {
                    deleteBtn.style.display = 'flex';
                };
                li.onmouseleave = () => {
                    deleteBtn.style.display = 'none';
                };

                // 🆕 保留原有的悬停样式（如果需要调整可以在这里修改）
                const conversationInfo = document.createElement('span');
                conversationInfo.className = 'conversation-info';
                conversationInfo.textContent = `${session.conversation_count}条`;
                conversationInfo.style.fontSize = '0.85em';
                conversationInfo.style.color = '#999';
                // 🆕 高亮状态下使用浅色文字
                if (li.classList.contains('active')) {
                    conversationInfo.style.color = 'rgba(255, 255, 255, 0.7)';
                }
                li.appendChild(conversationInfo);

                li.title = `${title}\n对话数: ${session.conversation_count}\n时间: ${new Date(session.updated_at).toLocaleString('zh-CN')}`;
                li.onclick = () => restoreSession(session);

                historyList.appendChild(li);
            });
        } else {
            const emptyTip = document.createElement('li');
            emptyTip.className = 'history-empty';
            emptyTip.textContent = '暂无历史记录';
            emptyTip.style.textAlign = 'center';
            emptyTip.style.color = '#999';
            emptyTip.style.padding = '20px';
            historyList.appendChild(emptyTip);
        }
    } catch (error) {
        console.error("加载历史记录失败:", error);
    }
}

// 🆕 删除会话确认和执行
async function confirmDeleteSession(sessionId, title) {
    const confirmed = confirm(`确定要删除会话"${title}"吗？\n此操作不可恢复。`);
    if (confirmed) {
        await deleteSession(sessionId);
    }
}

async function deleteSession(sessionId) {
    try {
        const response = await fetch(`${getApiUrl()}/history/session/${sessionId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            console.log('会话已删除:', sessionId);

            // 如果删除的是当前会话，重置UI
            if (currentSessionId === sessionId) {
                currentSessionId = null;
                messagesWrapper.innerHTML = '';
                messagesWrapper.appendChild(welcomeScreen);
                welcomeScreen.style.display = 'flex';
            }

            // 重新加载历史列表
            loadSavedHistory();

            // 显示成功提示
            if (window.showToast) {
                window.showToast('会话已删除', 'success');
            }
        } else {
            throw new Error('删除失败');
        }
    } catch (error) {
        console.error('删除会话失败:', error);
        if (window.showToast) {
            window.showToast('删除会话失败', 'error');
        } else {
            alert('删除会话失败');
        }
    }
}

function restoreSession(session) {
    currentSessionId = session.session_id;
    console.log("已恢复会话 ID:", currentSessionId);

    // 🆕 更新所有历史项的高亮状态
    const allHistoryItems = document.querySelectorAll('.history-item');
    allHistoryItems.forEach(item => {
        if (item.dataset.sessionId === currentSessionId) {
            item.classList.add('active');
            // 🆕 高亮状态下更新子元素样式
            const info = item.querySelector('.conversation-info');
            if (info) info.style.color = 'rgba(255, 255, 255, 0.7)';
        } else {
            item.classList.remove('active');
            // 🆕 移除高亮时恢复默认样式
            const info = item.querySelector('.conversation-info');
            if (info) info.style.color = '#999';
        }
    });

    resetChatUI();

    if (session.conversation && session.conversation.length > 0) {
        session.conversation.forEach(conv => {
            if (conv.user_content) {
                addMessage('user', conv.user_content);
            }

            if (conv.thinking_steps && conv.thinking_steps.length > 0) {
                renderThinkingSteps(conv.thinking_steps);
            }

            if (conv.ai_content) {
                addMessage('assistant', conv.ai_content);
            }
        });
    }
}

function renderThinkingSteps(steps) {
    if (!steps || steps.length === 0) return;

    const thinkingContainer = document.createElement('div');
    thinkingContainer.className = 'thinking-process';
    thinkingContainer.innerHTML = `
        <div class="thinking-header" onclick="toggleThinking(this)">
            <span class="thinking-toggle">▼</span>
            <span class="thinking-title">思考过程</span>
            <span class="thinking-icon">⚙️</span>
        </div>
        <div class="thinking-content"></div>
    `;

    const stepsContainer = thinkingContainer.querySelector('.thinking-content');

    steps.forEach(step => {
        const stepDiv = document.createElement('div');
        stepDiv.className = `thinking-step ${getStepClass(step.step_type)}`;

        const icon = getStepIcon(step.step_type);
        const title = step.title || '处理中';
        const description = step.description || '';

        stepDiv.innerHTML = `
            <span class="step-icon">${icon}</span>
            <div class="step-content">
                <div class="step-title">${title}</div>
                <div class="step-description">${description}</div>
            </div>
        `;

        stepsContainer.appendChild(stepDiv);
    });

    messagesWrapper.appendChild(thinkingContainer);
    scrollToBottom();
}

function resetChatUI() {
    messagesWrapper.innerHTML = '';
    hideWelcomeScreen();
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    clearAttachedFiles();
}

function startNewChat() {
    currentSessionId = null;
    console.log("开启新会话，Session ID 已重置");

    loadSavedHistory();

    messagesWrapper.innerHTML = '';
    messagesWrapper.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';

    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    messageInput.value = '';
    messageInput.focus();
    clearAttachedFiles();
}

// ============ 3. WebSocket 连接管理 ============
function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }

    ws = new WebSocket(getWsUrl());

    ws.onopen = () => {
        console.log('✅ WebSocket 连接成功');
        isConnected = true;
        reconnectAttempts = 0;
        updateStatus(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket 错误:', error);
        updateStatus(false);
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket 连接关闭');
        isConnected = false;
        updateStatus(false);

        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            setTimeout(connectWebSocket, delay);
        }
    };
}

function handleWebSocketMessage(payload) {
    console.log('📨 收到 WebSocket 消息:', payload);

    switch (payload.type) {
        case 'step':
            handleStepUpdate(payload);
            break;

        case 'token':
            const token = payload.content || (payload.data ? payload.data.content : '');
            if (token) handleTokenUpdate(token);
            break;

        case 'done':
            const finalMsg = (payload.data && payload.data.message)
                ? payload.data.message
                : payload.message;

            if (payload.data && payload.data.session_id) {
                currentSessionId = payload.data.session_id;
            }

            console.log('✅ 提取到最终消息:', finalMsg);
            handleDone(finalMsg);
            break;

        case 'error':
            const errMsg = payload.data ? payload.data.message : payload.message;
            handleError(errMsg);
            break;
    }
}

function updateStatus(connected) {
    if (statusIndicator) {
        if (connected) {
            statusIndicator.className = 'status-indicator connected';
            if (statusText) statusText.textContent = '已连接';
        } else {
            statusIndicator.className = 'status-indicator disconnected';
            if (statusText) statusText.textContent = '未连接';
        }
    }
}

// ============ 4. 步骤状态处理（集成到消息框内） ============
// 记录当前处理状态
let currentProcessingStatus = null;

function handleStepUpdate(payload) {
    hideWelcomeScreen();

    const stepData = payload.data || {};
    const stepType = stepData.step || 'processing';
    const title = stepData.title || '处理中';
    const description = stepData.description || '';

    console.log("处理步骤更新:", stepType, title);

    // 记录当前状态
    currentProcessingStatus = { stepType, title, description };

    // 如果已经有流式消息，更新其状态指示器
    if (currentStreamingAnswer) {
        updateStreamingStatus(stepType, title);
    } else {
        // 创建 AI 消息框并显示状态
        createStatusMessage(title, description, stepType);
    }
}

function createStatusMessage(title, description, stepType) {
    const statusDiv = document.createElement('div');
    statusDiv.className = 'message assistant processing';
    statusDiv.id = 'processingStatus';

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    // 根据步骤类型选择图标
    let icon = getStepIcon(stepType);
    let statusClass = getStepClass(stepType);

    statusDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar status-avatar ${statusClass}">${icon}</div>
            <div class="message-author">AI 助手</div>
            <div class="message-time">${time}</div>
        </div>
        <div class="message-content processing-content">
            <div class="processing-indicator">
                <div class="processing-dots">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
                <span class="processing-text">${title}</span>
            </div>
            ${description ? `<div class="processing-description">${escapeHtml(description)}</div>` : ''}
        </div>
    `;

    messagesWrapper.appendChild(statusDiv);
    scrollToBottom();
}

function updateStreamingStatus(stepType, title) {
    const statusEl = currentStreamingAnswer.querySelector('.processing-indicator .processing-text');
    if (statusEl) {
        statusEl.textContent = title;
    }
}

function getStepIcon(step) {
    if (!step || typeof step !== 'string') return '⚙️';
    const s = step.toLowerCase();

    if (s.includes('init') || s.includes('开始')) return '🚀';
    if (s.includes('tool_start') || s.includes('调用')) return '🔧';
    if (s.includes('tool_end') || s.includes('完成')) return '✅';
    if (s.includes('finish') || s.includes('结束')) return '🎯';
    if (s.includes('error')) return '❌';

    return '⚙️';
}

function getStepClass(step) {
    if (!step || typeof step !== 'string') return 'processing';
    const s = step.toLowerCase();
    if (s.includes('init') || s.includes('开始')) return 'starting';
    if (s.includes('tool_start') || s.includes('调用')) return 'tooling';
    if (s.includes('tool_end') || s.includes('完成')) return 'complete';
    if (s.includes('finish') || s.includes('结束')) return 'done';
    if (s.includes('error')) return 'error';
    return 'processing';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============ 5. 消息渲染与流式处理 ============
function hideWelcomeScreen() {
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
}

function addMessage(role, content) {
    hideWelcomeScreen();

    const div = document.createElement('div');
    div.className = `message ${role}`;

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    let innerHTML = '';

    if (role === 'user') {
        const textDiv = document.createElement('div');
        textDiv.textContent = content;
        innerHTML = `
            <div class="message-header">
                <div class="message-avatar"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>
                <div class="message-author">你</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content">${textDiv.innerHTML}</div>
        `;
    } else {
        const parsed = typeof marked !== 'undefined' ? marked.parse(content) : content;
        innerHTML = `
            <div class="message-header">
                <div class="message-avatar"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/><path d="M12 2a10 10 0 0 1 10 10h-10V2z"/></svg></div>
                <div class="message-author">AI 助手</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content">${parsed}</div>
        `;
    }

    div.innerHTML = innerHTML;
    messagesWrapper.appendChild(div);
    scrollToBottom();
}

function handleTokenUpdate(token) {
    // 收到 token 时，如果有状态消息，移除它
    const statusEl = document.getElementById('processingStatus');
    if (statusEl) {
        statusEl.remove();
    }

    // 清除当前处理状态
    currentProcessingStatus = null;

    if (!currentStreamingAnswer) {
        currentStreamingAnswer = document.createElement('div');
        currentStreamingAnswer.className = 'message assistant';
        const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

        currentStreamingAnswer.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">🤖</div>
                <div class="message-author">AI 助手</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content streaming" data-raw=""><span class="cursor-blink">|</span></div>
        `;
        messagesWrapper.appendChild(currentStreamingAnswer);
    }

    const contentDiv = currentStreamingAnswer.querySelector('.message-content');
    const cursor = contentDiv.querySelector('.cursor-blink');
    const currentRaw = contentDiv.dataset.raw || '';
    const newRaw = currentRaw + token;
    contentDiv.dataset.raw = newRaw;

    // 先移除光标再渲染
    if (cursor) cursor.remove();

    if (typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(newRaw) + '<span class="cursor-blink">|</span>';
    } else {
        contentDiv.textContent = newRaw;
    }

    scrollToBottom();
}

function handleDone(finalMessage) {
    console.log('🏁 handleDone 执行,finalMessage:', finalMessage);

    // 移除可能存在的状态消息
    const statusEl = document.getElementById('processingStatus');
    if (statusEl) {
        statusEl.remove();
    }
    currentProcessingStatus = null;

    if (currentStreamingAnswer) {
        const contentDiv = currentStreamingAnswer.querySelector('.message-content');
        contentDiv.classList.remove('streaming');
        if (finalMessage) {
            if (typeof marked !== 'undefined') {
                contentDiv.innerHTML = marked.parse(finalMessage);
            } else {
                contentDiv.textContent = finalMessage;
            }
        }
    }
    else if (finalMessage) {
        console.log('📝 没有流式框,手动添加最终消息');
        addMessage('assistant', finalMessage);
    } else {
        console.warn('⚠️ handleDone 被调用但没有消息内容,也没有流式框');
    }

    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;

    if (sendButton) sendButton.disabled = false;
    if (messageInput) {
        messageInput.disabled = false;
        messageInput.focus();
    }

    if (typeof loadSavedHistory === 'function') {
        loadSavedHistory();
    }
}

function handleError(msg) {
    addMessage('assistant', '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> 错误: ' + msg);
    isProcessing = false;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
}

// ============ 6. 文件上传功能 ============
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const iconMap = {
        'pdf': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        'doc': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
        'docx': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
        'txt': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
        'md': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>',
        'markdown': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>',
        'csv': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/></svg>'
    };
    return iconMap[ext] || '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>';
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

function clearAttachedFiles() {
    attachedFiles = [];
    renderFileAttachments();
}

// ============ 7. 发送与交互逻辑 ============
async function sendMessage(text = null) {
    const message = text || messageInput.value.trim();

    if ((!message && attachedFiles.length === 0) || !isConnected || isProcessing) {
        if (!isConnected && window.showToast) {
            window.showToast("未连接到服务器", "error");
        }
        return;
    }

    if (attachedFiles.length > 0) {
        // 带附件的上传 (使用 HTTP POST)
        try {
            const formData = new FormData();
            formData.append('message', message);

            if (currentSessionId) {
                formData.append('session_id', currentSessionId);
            }

            attachedFiles.forEach((file, index) => {
                formData.append('files', file);
            });

            let userMessage = message;
            if (attachedFiles.length > 0) {
                const fileNames = attachedFiles.map(f => f.name).join(', ');
                userMessage += `\n\n📎 附件: ${fileNames}`;
            }
            addMessage('user', userMessage);

            isProcessing = true;
            sendButton.disabled = true;
            messageInput.disabled = true;

            const response = await fetch(`${getApiUrl()}/chat/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('文件上传失败');
            }

            const result = await response.json();

            clearAttachedFiles();
            console.log("收到附件上传回复:", result);

            if (result.session_id) {
                currentSessionId = result.session_id;
                console.log("会话 ID 更新为:", currentSessionId);
            }

            loadSavedHistory();

            if (result.message) {
                handleDone(result.message);
            } else {
                isProcessing = false;
                sendButton.disabled = false;
                messageInput.disabled = false;
            }

        } catch (error) {
            console.error('文件上传错误:', error);
            handleError('文件上传失败: ' + error.message);
        }
    } else {
        // 普通对话 (使用 WebSocket)
        addMessage('user', message);

        const payload = { message };
        if (currentSessionId) {
            payload.session_id = currentSessionId;
        }

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
        if (messagesArea) {
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }
    });
}

function autoResizeTextarea() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
}

function attachQuickPromptListeners() {
    const cards = document.querySelectorAll('.quick-prompt-card');
    cards.forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            sendMessage(prompt);
        });
    });
}

// ============ 8. 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    initDOM();

    if (sendButton) sendButton.addEventListener('click', () => sendMessage());

    if (messageInput) {
        messageInput.addEventListener('input', autoResizeTextarea);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);
    if (sidebarNewChatBtn) sidebarNewChatBtn.addEventListener('click', startNewChat);

    if (attachButton) {
        attachButton.addEventListener('click', () => {
            chatFileInput.click();
        });
    }

    if (chatFileInput) {
        chatFileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);

            if (attachedFiles.length + files.length > 10) {
                if (window.showToast) {
                    window.showToast('最多只能上传10个文件', 'error');
                } else {
                    alert('最多只能上传10个文件');
                }
                chatFileInput.value = '';
                return;
            }

            const maxSize = 10 * 1024 * 1024;
            for (let file of files) {
                if (file.size > maxSize) {
                    if (window.showToast) {
                        window.showToast(`文件 ${file.name} 超过10MB限制`, 'error');
                    } else {
                        alert(`文件 ${file.name} 超过10MB限制`);
                    }
                    chatFileInput.value = '';
                    return;
                }
            }

            attachedFiles.push(...files);
            renderFileAttachments();
            chatFileInput.value = '';
        });
    }

    window.toggleThinking = function(header) {
        const content = header.nextElementSibling;
        const toggle = header.querySelector('.thinking-toggle');
        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggle.style.transform = 'rotate(0deg)';
        } else {
            content.style.display = 'none';
            toggle.style.transform = 'rotate(-90deg)';
        }
    };

    attachQuickPromptListeners();
    connectWebSocket();
    loadSavedHistory();

    console.log('✅ chat.js 已加载');
});
