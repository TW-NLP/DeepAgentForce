/**
 * 配置管理模块
 * 负责加载、保存和验证模型配置
 * 适配扁平化配置结构 (无嵌套组)
 */

const API_BASE = 'http://localhost:8000/api';

// DOM ID 到 配置文件 Key 的映射
// 格式: { HTML元素ID : 后端配置Key }
const FIELD_MAPPING = {
    // LLM 配置
    'llmApiKey': 'LLM_API_KEY',
    'llmUrl': 'LLM_URL',
    'llmModel': 'LLM_MODEL',
    
    // 搜索配置
    'tavilyApiKey': 'TAVILY_API_KEY',
    
    // Firecrawl 配置
    'firecrawlApiKey': 'FIRECRAWL_API_KEY',
    'firecrawlUrl': 'FIRECRAWL_URL',
    
    // Embedding 配置
    'embeddingApiKey': 'EMBEDDING_API_KEY',
    'embeddingUrl': 'EMBEDDING_URL',
    'embeddingModel': 'EMBEDDING_MODEL'
};

// ==================== 工具函数 ====================

// 安全调用全局 Toast
function safeShowToast(message, type) {
    if (typeof window.showToast === 'function') {
        window.showToast(message, type);
    } else {
        console.log(`[${type}] ${message}`);
        // 如果没有全局 toast，可以使用简单的 alert 作为回退，或者什么都不做
        if (type === 'error') alert(message);
    }
}

// ==================== 加载配置 ====================

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 确保 data.config 存在
        if (!data.success || !data.config) {
            console.log('⚠️ 未找到有效配置或配置为空');
            return;
        }

        const config = data.config; // 扁平化对象

        // 遍历映射表填充数据
        for (const [domId, configKey] of Object.entries(FIELD_MAPPING)) {
            const element = document.getElementById(domId);
            if (element) {
                // 如果配置中有值则填充，否则保持原样或清空
                element.value = config[configKey] || '';
            }
        }
        
        console.log('✅ 配置加载完成');
    } catch (error) {
        console.error('❌ 加载配置失败:', error);
        safeShowToast('加载配置失败，请检查后端服务', 'error');
    }
}

// ==================== 保存配置 ====================

async function saveConfig() {
    const saveButton = document.getElementById('saveConfigBtn');
    if (saveButton) {
        saveButton.disabled = true;
        saveButton.textContent = '保存中...';
    }
    
    try {
        // 1. 收集数据
        const configToSave = {};
        for (const [domId, configKey] of Object.entries(FIELD_MAPPING)) {
            const element = document.getElementById(domId);
            if (element && element.value.trim() !== '') {
                configToSave[configKey] = element.value.trim();
            }
        }
        
        // 2. 验证必填项 (根据需求调整)
        const requiredKeys = ['LLM_API_KEY', 'LLM_URL', 'LLM_MODEL'];
        const missing = requiredKeys.filter(key => !configToSave[key]);
        
        if (missing.length > 0) {
            throw new Error(`缺少必填项: ${missing.join(', ')}`);
        }
        
        // 3. 发送请求
        const response = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configToSave)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || '保存请求失败');
        }
        
        const result = await response.json();
        
        safeShowToast('配置已保存！新的对话将使用新配置', 'success');
        console.log('✅ 保存成功:', result);
        
    } catch (error) {
        console.error('❌ 保存配置失败:', error);
        safeShowToast(error.message, 'error');
    } finally {
        if (saveButton) {
            saveButton.disabled = false;
            saveButton.textContent = '保存配置';
        }
    }
}

// ==================== 初始化 ====================

function initConfigPage() {
    // 绑定保存按钮
    const saveButton = document.getElementById('saveConfigBtn');
    if (saveButton) {
        // 移除旧的监听器（防止重复绑定，虽然后面是直接执行，但习惯上要注意）
        saveButton.removeEventListener('click', saveConfig);
        saveButton.addEventListener('click', saveConfig);
    }
    
    // 加载数据
    loadConfig();
}

// 确保 DOM 加载完毕后执行
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initConfigPage);
} else {
    initConfigPage();
}

// 暴露给全局 (方便控制台调试或 HTML onclick 调用)
window.loadConfig = loadConfig;
window.saveConfig = saveConfig;