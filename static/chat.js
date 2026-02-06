/**
 * 对话功能 JavaScript - 完整版
 * 包含：WebSocket 流式对话、历史记录侧边栏加载、思考过程展示
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

// DOM 元素引用
const messagesWrapper = document.getElementById('messagesWrapper');
const messagesArea = document.getElementById('messagesArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const historyList = document.getElementById('historyList'); // 侧边栏列表
const newChatBtn = document.getElementById('newChatBtn'); // 顶部新建按钮
const sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn'); // 侧边栏新建按钮
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// ============ 1. 历史记录加载与管理 ============
async function loadSavedHistory() {
    try {
        console.log("正在加载历史记录...");
        const response = await fetch(`${API_URL}/history/saved`);  // ← 修复这里
        
        if (!response.ok) {
            console.warn("无法连接到历史记录接口");
            return;
        }

        const data = await response.json();
        
        // 清空列表
        if (historyList) {
            historyList.innerHTML = '';
        }

        // 适配新的数据结构：sessions
        if (data.success && Array.isArray(data.sessions) && data.sessions.length > 0) {
            // 按更新时间倒序排列
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
 * 恢复显示某一段历史对话
 */
function restoreSession(session) {
    // 1. 清空当前屏幕
    resetChatUI();

    // 2. 遍历显示所有对话
    if (session.conversation && session.conversation.length > 0) {
        session.conversation.forEach(msg => {
            // 显示用户提问
            if (msg.user_content) {
                addMessage('user', msg.user_content);
            }
            // 显示 AI 回答
            if (msg.ai_content) {
                addMessage('assistant', msg.ai_content);
            }
        });
    }
}
/**
 * 重置聊天界面 (清空消息，显示欢迎页)
 * 但这里我们实际上是清空消息，隐藏欢迎页(如果有新消息)
 */
function resetChatUI() {
    messagesWrapper.innerHTML = '';
    // 隐藏欢迎页 (因为要显示消息了)
    hideWelcomeScreen();
    // 重置状态
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
}

/**
 * 完全重置为初始状态 (点击新建对话时)
 */
function startNewChat() {
    messagesWrapper.innerHTML = '';
    // 重新把欢迎页放回去
    messagesWrapper.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';
    
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    messageInput.value = '';
    messageInput.focus();
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
            // 兼容 token 可能的位置
            const token = payload.content || (payload.data ? payload.data.content : '');
            if (token) handleTokenUpdate(token);
            break;
            
        case 'done':
            // 【关键修复】从 payload.data.message 提取最终文本
            // 如果 payload.data 不存在，尝试直接读取 payload.message
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
function handleStepUpdate(payload) {
    hideWelcomeScreen();

    // 🔍 这里的 payload 是整个 WebSocket 消息对象
    // 我们需要取里面的 data 字段
    const stepData = payload.data || {}; 
    
    // 提取 step 类型
    const stepType = stepData.step || 'processing';

    console.log("处理步骤更新:", stepType); 

    // 如果还没有思考容器，创建一个
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
    
    // ✅ 正确传递 stepType
    stepDiv.className = `thinking-step ${getStepClass(stepType)}`;
    
    const icon = getStepIcon(stepType); 
    const title = stepData.title || '处理中';
    
    // 处理 description 可能是对象的情况
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

function handleDone(finalMessage) {
    console.log('🏁 handleDone 执行，finalMessage:', finalMessage);
    
    // 情况 A: 之前有流式输出框 (currentStreamingAnswer 存在)
    if (currentStreamingAnswer) {
        const contentDiv = currentStreamingAnswer.querySelector('.message-content');
        contentDiv.classList.remove('streaming');
        // 确保最终内容完整（防止流式丢包，用最终结果覆盖一次）
        if (finalMessage) {
             if (typeof marked !== 'undefined') {
                contentDiv.innerHTML = marked.parse(finalMessage);
            } else {
                contentDiv.textContent = finalMessage;
            }
        }
    } 
    // 情况 B: 之前没有流式输出 (比如这次只有思考过程，没有产生 token，直接 done)
    // 必须手动添加一条 AI 消息
    else if (finalMessage) {
        console.log('📝 没有流式框，手动添加最终消息');
        addMessage('assistant', finalMessage);
    } else {
        console.warn('⚠️ handleDone 被调用但没有消息内容，也没有流式框');
    }
    
    // 清理状态
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    
    // 恢复按钮状态
    if (sendButton) sendButton.disabled = false;
    if (messageInput) {
        messageInput.disabled = false;
        messageInput.focus();
    }
    
    // 刷新历史记录列表
    if (typeof loadSavedHistory === 'function') {
        loadSavedHistory();
    }
}
// ============ 3. 消息渲染与流式处理 ============

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
        // 用户消息，简单文本转义
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
        // AI 消息，Markdown 解析
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

function handleStepUpdate(payload) {
    // 1. 隐藏欢迎页
    hideWelcomeScreen();

    console.log("正在处理 Step 数据:", payload); // 调试日志

    // 🔥 核心修正点：必须先从 payload 中取出 data 字段
    // payload 结构是: { type: 'step', data: { step: 'init', title: '...' } }
    const stepData = payload.data || {}; 
    
    // 现在 stepData.step 才是真正的 "init"
    const stepType = stepData.step || 'processing';

    // 2. 如果还没有思考容器，创建一个
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
    
    // 3. 创建步骤条目
    const stepDiv = document.createElement('div');
    
    // 🔥 修正点：这里传入提取好的 stepType ('init')，而不是 undefined
    stepDiv.className = `thinking-step ${getStepClass(stepType)}`;
    
    const icon = getStepIcon(stepType);
    const title = stepData.title || '处理中';
    
    // 处理 description 可能是对象的情况
    let description = stepData.description || '';
    if (typeof description === 'object') {
        try {
            description = JSON.stringify(description);
        } catch(e) {
            description = "详细信息...";
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
    // 🛡️ 防御代码
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
    // 🛡️ 防御代码：如果 step 是 undefined、null 或者不是字符串，直接返回空字符串
    if (!step || typeof step !== 'string') {
        console.warn("getStepClass 接收到了无效参数:", step); // 方便调试
        return '';
    }
    
    const s = step.toLowerCase();
    if (s.includes('analyzing')) return 'analyzing';
    if (s.includes('plan')) return 'planning';
    if (s.includes('chat')) return 'chatting';
    if (s.includes('error')) return 'error';
    return '';
}

// 处理文本流 (Token)
function handleTokenUpdate(token) {
    if (!currentStreamingAnswer) {
        // 创建新的 AI 回复框
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
    // 获取当前暂存的原始文本
    const currentRaw = contentDiv.dataset.raw || '';
    const newRaw = currentRaw + token;
    contentDiv.dataset.raw = newRaw;
    
    // 实时解析 Markdown
    if (typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(newRaw);
    } else {
        contentDiv.textContent = newRaw;
    }
    
    scrollToBottom();
}


// 处理错误 (Error)
function handleError(msg) {
    addMessage('assistant', `❌ 错误: ${msg}`);
    isProcessing = false;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
}

// ============ 4. 发送与交互逻辑 ============

function sendMessage(text = null) {
    const message = text || messageInput.value.trim();
    
    if (!message || !isConnected || isProcessing) {
        if (!isConnected) showToast("未连接到服务器", "error");
        return;
    }

    // 1. 显示用户消息
    addMessage('user', message);
    
    // 2. 发送 WebSocket
    ws.send(JSON.stringify({ message }));
    
    // 3. UI 状态更新
    if (!text) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
    
    isProcessing = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
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

// ============ 5. 初始化绑定 ============

// 绑定发送按钮
if (sendButton) sendButton.addEventListener('click', () => sendMessage());

// 绑定输入框回车
if (messageInput) {
    messageInput.addEventListener('input', autoResizeTextarea);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// 绑定新建对话按钮 (Header 和 侧边栏)
if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);
if (sidebarNewChatBtn) sidebarNewChatBtn.addEventListener('click', startNewChat);

// 启动
attachQuickPromptListeners();
connectWebSocket();
loadSavedHistory(); // 页面加载时自动获取历史记录