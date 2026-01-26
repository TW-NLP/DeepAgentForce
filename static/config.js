/**
 * 配置管理模块
 * 负责加载、保存和验证模型配置
 */

const API_BASE = 'http://localhost:8000/api';

// 配置字段映射
const CONFIG_FIELDS = {
    // LLM 配置
    llmApiKey: { key: 'LLM_API_KEY', group: 'llm_config' },
    llmUrl: { key: 'LLM_URL', group: 'llm_config' },
    llmModel: { key: 'LLM_MODEL', group: 'llm_config' },
    // 搜索配置
    tavilyApiKey: { key: 'TAVILY_API_KEY', group: 'search_config' },
    // Firecrawl 配置
    firecrawlApiKey: { key: 'FIRECRAWL_API_KEY', group: 'firecrawl_config' },
    firecrawlUrl: { key: 'FIRECRAWL_URL', group: 'firecrawl_config' },
    // Embedding 配置
    embeddingApiKey: { key: 'EMBEDDING_API_KEY', group: 'embedding_config' },
    embeddingUrl: { key: 'EMBEDDING_URL', group: 'embedding_config' },
    embeddingModel: { key: 'EMBEDDING_MODEL', group: 'embedding_config' }
};

// ==================== 加载配置 ====================

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        if (!response.ok) {
            throw new Error('加载配置失败');
        }
        
        const data = await response.json();
        
        if (!data.success || !data.config) {
            console.log('⚠️ 未找到已保存的配置');
            return;
        }
        
        // 填充表单
        for (const [fieldId, fieldInfo] of Object.entries(CONFIG_FIELDS)) {
            const element = document.getElementById(fieldId);
            if (!element) continue;
            
            const { key, group } = fieldInfo;
            
            // 从对应的配置组中获取值
            if (data.config[group] && data.config[group][key]) {
                element.value = data.config[group][key];
            }
        }
        
        console.log('✅ 配置加载成功');
    } catch (error) {
        console.error('❌ 加载配置失败:', error);
        if (typeof showToast === 'function') {
            showToast('加载配置失败', 'error');
        }
    }
}

// ==================== 保存配置 ====================

async function saveConfig() {
    const saveButton = document.getElementById('saveConfigBtn');
    saveButton.disabled = true;
    
    try {
        // 收集配置数据
        const config = {};
        
        for (const [fieldId, fieldInfo] of Object.entries(CONFIG_FIELDS)) {
            const element = document.getElementById(fieldId);
            if (element && element.value.trim()) {
                config[fieldInfo.key] = element.value.trim();
            }
        }
        
        // 验证必填项
        const requiredFields = ['LLM_API_KEY', 'LLM_URL', 'LLM_MODEL'];
        const missingFields = requiredFields.filter(field => !config[field]);
        
        if (missingFields.length > 0) {
            if (typeof showToast === 'function') {
                showToast('请填写必填项：LLM API Key, URL 和模型名称', 'error');
            }
            saveButton.disabled = false;
            return;
        }
        
        // 发送保存请求
        const response = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '保存配置失败');
        }
        
        const result = await response.json();
        
        if (typeof showToast === 'function') {
            showToast('配置保存成功！部分配置需要重启服务才能生效', 'success');
        }
        console.log('✅ 配置保存成功:', result);
        
    } catch (error) {
        console.error('❌ 保存配置失败:', error);
        if (typeof showToast === 'function') {
            showToast(`保存失败: ${error.message}`, 'error');
        }
    } finally {
        saveButton.disabled = false;
    }
}

// ==================== 初始化配置页面 ====================

function initConfigPage() {
    const saveButton = document.getElementById('saveConfigBtn');
    if (saveButton) {
        saveButton.addEventListener('click', saveConfig);
    }
    
    // 页面加载时自动加载配置
    loadConfig();
}

// DOM 加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initConfigPage);
} else {
    initConfigPage();
}

// 导出函数供外部调用
window.loadConfig = loadConfig;
window.saveConfig = saveConfig;