/**
 * 对话功能 JavaScript - 优化版
 * 新增：历史记录恢复时显示思考过程
 */

const WS_URL = 'ws://localhost:8000/ws/stream';
const API_URL = 'http://localhost:8000/api';

let ws = null;
let isConnected = false;
let isProcessing = false;
let currentThinkingContainer = null;
let currentStreamingAnswer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// 📎 文件上传相关变量
let attachedFiles = [];

// DOM 元素引用
const messagesWrapper = document.getElementById('messagesWrapper');
const messagesArea = document.getElementById('messagesArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const attachButton = document.getElementById('attachButton');
const chatFileInput = document.getElementById('chatFileInput');
const fileAttachmentsContainer = document.getElementById('fileAttachments');
const historyList = document.getElementById('historyList');
const newChatBtn = document.getElementById('newChatBtn');
const sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// ============ 1. 历史记录加载与管理 ============
async function loadSavedHistory() {
    try {
        console.log("正在加载历史记录...");
        const response = await fetch(`${API_URL}/history/saved`);
        
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
                
                let title = session.title || '新对话';
                
                if (title === '新对话' && session.conversation && session.conversation.length > 0) {
                    const firstMsg = session.conversation[0].user_content;
                    if (firstMsg) {
                        title = firstMsg.length > 20 
                            ? firstMsg.substring(0, 20) + '...' 
                            : firstMsg;
                    }
                }
                
                li.textContent = title;
                
                const conversationInfo = document.createElement('span');
                conversationInfo.className = 'conversation-info';
                conversationInfo.textContent = ` (${session.conversation_count}条)`;
                conversationInfo.style.fontSize = '0.85em';
                conversationInfo.style.color = '#999';
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

/**
 * 🆕 恢复会话时显示完整内容（包括思考过程）
 */
function restoreSession(session) {
    resetChatUI();

    if (session.conversation && session.conversation.length > 0) {
        session.conversation.forEach(conv => {
            // 1. 显示用户消息
            if (conv.user_content) {
                addMessage('user', conv.user_content);
            }
            
            // 2. 🔥 显示思考过程（如果有）
            if (conv.thinking_steps && conv.thinking_steps.length > 0) {
                renderThinkingSteps(conv.thinking_steps);
            }
            
            // 3. 显示 AI 回答
            if (conv.ai_content) {
                addMessage('assistant', conv.ai_content);
            }
        });
    }
}

/**
 * 🆕 渲染历史记录中的思考过程
 */
function renderThinkingSteps(steps) {
    if (!steps || steps.length === 0) return;
    
    // 创建思考容器
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
    
    // 渲染每个步骤
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

// ============ 2. WebSocket 连接管理 ============

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }

    ws = new WebSocket(WS_URL);

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

// ============ 3. 思考过程处理 ============

function handleStepUpdate(payload) {
    hideWelcomeScreen();

    const stepData = payload.data || {}; 
    const stepType = stepData.step || 'processing';

    console.log("处理步骤更新:", stepType); 

    if (!currentThinkingContainer) {
        currentThinkingContainer = document.createElement('div');
        currentThinkingContainer.className = 'thinking-process';
        currentThinkingContainer.innerHTML = `
            <div class="thinking-header" onclick="toggleThinking(this)">
                <span class="thinking-toggle">▼</span>
                <span class="thinking-title">思考过程</span>
                <span class="thinking-icon">⚙️</span>
            </div>
            <div class="thinking-content"></div>
        `;
        messagesWrapper.appendChild(currentThinkingContainer);
    }

    const stepsContainer = currentThinkingContainer.querySelector('.thinking-content');
    
    const stepDiv = document.createElement('div');
    stepDiv.className = `thinking-step ${getStepClass(stepType)}`;
    
    const icon = getStepIcon(stepType); 
    const title = stepData.title || '处理中';
    
    let description = stepData.description || '';
    if (typeof description === 'object') {
        try {
            description = JSON.stringify(description);
        } catch(e) {
            description = "复杂数据";
        }
    }
    
    stepDiv.innerHTML = `
        <span class="step-icon">${icon}</span>
        <div class="step-content">
            <div class="step-title">${title}</div>
            <div class="step-description">${description}</div>
        </div>
    `;
    
    stepsContainer.appendChild(stepDiv);
    scrollToBottom();
}

function getStepIcon(step) {
    if (!step || typeof step !== 'string') {
        return '⚙️';
    }

    const s = step.toLowerCase();
    
    if (s.includes('init') || s.includes('开始')) return '🤔';
    if (s.includes('tool_start') || s.includes('调用')) return '🔧';
    if (s.includes('tool_end') || s.includes('完成')) return '✅';
    if (s.includes('finish') || s.includes('结束')) return '🎯';
    if (s.includes('error')) return '❌';
    
    return '⚙️';
}

function getStepClass(step) {
    if (!step || typeof step !== 'string') {
        console.warn("getStepClass 接收到了无效参数:", step);
        return '';
    }
    
    const s = step.toLowerCase();
    if (s.includes('analyzing')) return 'analyzing';
    if (s.includes('plan')) return 'planning';
    if (s.includes('chat')) return 'chatting';
    if (s.includes('error')) return 'error';
    return '';
}

// ============ 4. 消息渲染与流式处理 ============

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
                <div class="message-avatar">👤</div>
                <div class="message-author">你</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content">${textDiv.innerHTML}</div>
        `;
    } else {
        const parsed = typeof marked !== 'undefined' ? marked.parse(content) : content;
        innerHTML = `
            <div class="message-header">
                <div class="message-avatar">🤖</div>
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
            <div class="message-content streaming" data-raw=""></div>
        `;
        messagesWrapper.appendChild(currentStreamingAnswer);
    }
    
    const contentDiv = currentStreamingAnswer.querySelector('.message-content');
    const currentRaw = contentDiv.dataset.raw || '';
    const newRaw = currentRaw + token;
    contentDiv.dataset.raw = newRaw;
    
    if (typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(newRaw);
    } else {
        contentDiv.textContent = newRaw;
    }
    
    scrollToBottom();
}

function handleDone(finalMessage) {
    console.log('🏁 handleDone 执行,finalMessage:', finalMessage);
    
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
    addMessage('assistant', `❌ 错误: ${msg}`);
    isProcessing = false;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
}

// ============ 5. 文件上传功能 ============

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
        'pdf': '📄',
        'doc': '📝',
        'docx': '📝',
        'txt': '📃',
        'md': '📋',
        'markdown': '📋',
        'csv': '📊'
    };
    return iconMap[ext] || '📎';
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

window.removeAttachment = removeAttachment;

// ============ 6. 发送与交互逻辑 ============

async function sendMessage(text = null) {
    const message = text || messageInput.value.trim();
    
    if ((!message && attachedFiles.length === 0) || !isConnected || isProcessing) {
        if (!isConnected && window.showToast) {
            window.showToast("未连接到服务器", "error");
        }
        return;
    }

    if (attachedFiles.length > 0) {
        try {
            const formData = new FormData();
            formData.append('message', message);
            attachedFiles.forEach((file, index) => {
                formData.append('files', file);
            });

            let userMessage = message;
            if (attachedFiles.length > 0) {
                const fileNames = attachedFiles.map(f => f.name).join(', ');
                userMessage += `\n\n📎 附件: ${fileNames}`;
            }
            addMessage('user', userMessage);
            
            const response = await fetch(`${API_URL}/chat/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('文件上传失败');
            }
            
            const result = await response.json();
            
            clearAttachedFiles();
            
            isProcessing = true;
            sendButton.disabled = true;
            messageInput.disabled = true;
            
        } catch (error) {
            console.error('文件上传错误:', error);
            if (window.showToast) {
                window.showToast('文件上传失败: ' + error.message, 'error');
            }
            return;
        }
    } else {
        addMessage('user', message);
        ws.send(JSON.stringify({ message }));
        
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

// ============ 7. 初始化绑定 ============

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

attachQuickPromptListeners();
connectWebSocket();
loadSavedHistory();