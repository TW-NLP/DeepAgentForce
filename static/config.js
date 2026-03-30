/**
 * 配置管理模块
 * 统一管理前后端地址配置
 * 配置来源优先级：
 * 1. URL 参数 ?api=xxx
 * 2. 后端 /api/server_info 接口
 * 3. 本地默认配置
 */

// ==================== 本地默认配置 ====================
const DEFAULT_CONFIG = {
    API_HOST: '127.0.0.1',
    API_PORT: 8000,
    FRONTEND_HOST: '127.0.0.1',
    FRONTEND_PORT: 8080,
};

// ==================== 全局配置对象 ====================
const CONFIG = {
    ...DEFAULT_CONFIG,
    _initialized: false,
    _apiBase: null,
    _wsBase: null,
};

// ==================== 配置获取函数 ====================

/**
 * 从 URL 参数或本地存储获取 API 地址
 */
function getApiBase() {
    // 1. 优先使用 URL 参数
    const urlParams = new URLSearchParams(window.location.search);
    const apiParam = urlParams.get('api');
    if (apiParam) {
        return apiParam.replace(/\/$/, ''); // 去除末尾斜杠
    }

    // 2. 返回缓存的 API 地址
    if (CONFIG._apiBase) {
        return CONFIG._apiBase;
    }

    // 3. 返回默认地址（不含 /api，由各路由自行添加）
    return `http://${CONFIG.API_HOST}:${CONFIG.API_PORT}`;
}

/**
 * 获取 WebSocket 地址
 */
function getWsBase() {
    // 1. 优先使用 URL 参数
    const urlParams = new URLSearchParams(window.location.search);
    const apiParam = urlParams.get('api');
    if (apiParam) {
        return apiParam.replace(/\/$/, '') + '/ws/stream';
    }

    // 2. 返回缓存的 WS 地址
    if (CONFIG._wsBase) {
        return CONFIG._wsBase;
    }

    // 3. 返回默认地址
    return `ws://${CONFIG.API_HOST}:${CONFIG.API_PORT}/ws/stream`;
}

/**
 * 获取完整的 API URL
 */
function getApiUrl() {
    // 1. 优先使用 URL 参数
    const urlParams = new URLSearchParams(window.location.search);
    const apiParam = urlParams.get('api');
    if (apiParam) {
        return apiParam.replace(/\/$/, ''); // 去除末尾斜杠
    }

    // 2. 返回缓存的 API 地址
    if (CONFIG._apiBase) {
        return CONFIG._apiBase;
    }

    // 3. 返回默认地址（始终包含 /api，与后端 server_info 保持一致）
    return `http://${CONFIG.API_HOST}:${CONFIG.API_PORT}/api`;
}

/**
 * 获取 WebSocket URL
 * 🆕 多租户：自动携带 access_token 用于后端提取 tenant_id
 */
function getWsUrl() {
    let base = getWsBase();
    
    // 🆕 从 auth 模块获取 token
    const token = window.auth && window.auth.getAccessToken ? window.auth.getAccessToken() : null;
    if (token) {
        const separator = base.includes('?') ? '&' : '?';
        base += `${separator}token=${encodeURIComponent(token)}`;
    }
    
    return base;
}

// ==================== 服务器信息同步 ====================

/**
 * 从后端获取服务器配置并更新本地配置
 */
async function syncServerConfig() {
    try {
        // 先用默认地址尝试获取
        const defaultUrl = `http://${DEFAULT_CONFIG.API_HOST}:${DEFAULT_CONFIG.API_PORT}/api`;
        const response = await fetch(`${defaultUrl}/server_info`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const serverInfo = await response.json();

        // 更新配置
        CONFIG.API_HOST = serverInfo.host || DEFAULT_CONFIG.API_HOST;
        CONFIG.API_PORT = serverInfo.port || DEFAULT_CONFIG.API_PORT;
        CONFIG._apiBase = serverInfo.api_base || `${defaultUrl}`;
        CONFIG._wsBase = serverInfo.ws_base || `ws://${DEFAULT_CONFIG.API_HOST}:${DEFAULT_CONFIG.API_PORT}/ws/stream`;
        CONFIG.FRONTEND_HOST = serverInfo.frontend_host || DEFAULT_CONFIG.FRONTEND_HOST;
        CONFIG.FRONTEND_PORT = serverInfo.frontend_port || DEFAULT_CONFIG.FRONTEND_PORT;
        CONFIG._initialized = true;

        console.log('✅ 服务器配置同步成功:', {
            api_base: CONFIG._apiBase,
            ws_base: CONFIG._wsBase,
            frontend_base: serverInfo.frontend_base
        });

        return true;
    } catch (error) {
        console.warn('⚠️ 无法获取服务器配置，使用默认配置:', error.message);
        return false;
    }
}

// ==================== 配置加载/保存 ====================

async function loadConfig() {
    try {
        // 🆕 使用 authFetch（用于后端按租户返回配置）
        const response = await authFetch(`${getApiUrl()}/config`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.success && data.config ? data.config : {};
    } catch (e) {
        console.error('加载配置失败:', e);
        return null;
    }
}

async function saveConfig(formData) {
    try {
        // 🆕 使用 authFetch（用于后端按租户保存配置）
        const response = await authFetch(`${getApiUrl()}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (e) {
        console.error('保存配置失败:', e);
        throw e;
    }
}

// ==================== UI 绑定 ====================

const FIELD_MAPPING = {
    'llmApiKey': 'LLM_API_KEY',
    'llmUrl': 'LLM_URL',
    'llmModel': 'LLM_MODEL',
    'tavilyApiKey': 'TAVILY_API_KEY',
    'firecrawlApiKey': 'FIRECRAWL_API_KEY',
    'firecrawlUrl': 'FIRECRAWL_URL',
    'embeddingApiKey': 'EMBEDDING_API_KEY',
    'embeddingUrl': 'EMBEDDING_URL',
    'embeddingModel': 'EMBEDDING_MODEL'
};

async function populateConfigFields() {
    const config = await loadConfig();
    if (!config) return;
    for (const [elementId, configKey] of Object.entries(FIELD_MAPPING)) {
        const element = document.getElementById(elementId);
        if (element && config[configKey]) {
            element.value = config[configKey];
        }
    }
}

async function saveConfigFromForm() {
    const formData = {};
    for (const [elementId, configKey] of Object.entries(FIELD_MAPPING)) {
        const element = document.getElementById(elementId);
        if (element) {
            formData[configKey] = element.value;
        }
    }
    return await saveConfig(formData);
}

// ==================== 暴露全局函数 ====================

// 基础 URL 函数
window.getApiBase = getApiBase;
window.getWsBase = getWsBase;
window.getApiUrl = getApiUrl;
window.getWsUrl = getWsUrl;

// 配置管理函数
window.loadConfig = loadConfig;
window.saveConfig = saveConfig;
window.populateConfigFields = populateConfigFields;
window.saveConfigFromForm = saveConfigFromForm;
window.syncServerConfig = syncServerConfig;

// 导出配置对象供其他模块使用
window.CONFIG = CONFIG;

// 初始化：自动同步服务器配置
(async function initConfig() {
    await syncServerConfig();
    console.log('✅ config.js 已加载');
})();
