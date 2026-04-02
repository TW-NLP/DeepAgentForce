/**
 * 配置管理模块
 * 统一管理前后端地址配置
 * 配置来源优先级：
 * 1. URL 参数 ?api=xxx
 * 2. 后端 /api/server_info 接口
 * 3. 本地默认配置
 */

// ==================== 本地默认配置 ====================
// 注意：这些仅作为最后回退，初始化时会优先从浏览器 URL 自动获取
const DEFAULT_CONFIG = {
    API_HOST: '',  // 空字符串，初始化时自动从浏览器 URL 获取
    API_PORT: 8000,
    FRONTEND_HOST: '',  // 空字符串，初始化时自动从浏览器 URL 获取
    FRONTEND_PORT: '',
};

// ==================== 全局配置对象 ====================
const CONFIG = {
    ...DEFAULT_CONFIG,
    _initialized: false,
    _apiBase: null,
    _wsBase: null,
    _frontendBase: null,
};

// ==================== 从浏览器 URL 自动检测地址 ====================

/**
 * 从浏览器 URL（window.location）自动解析前端和后端地址
 * 部署时无需修改代码，访问 http://<部署服务器IP>:8080 即可
 */
function autoDetectFromBrowser() {
    const protocol = window.location.protocol;  // 'http:' 或 'https:'
    const hostname = window.location.hostname;  // 浏览器地址栏的 IP 或域名
    const port = window.location.port;          // 浏览器地址栏的端口

    // 前端地址：直接使用浏览器当前地址
    CONFIG.FRONTEND_HOST = hostname;
    CONFIG.FRONTEND_PORT = port;

    // 后端地址：假设后端与前端同服务器，端口为 8000
    CONFIG.API_HOST = hostname;
    CONFIG.API_PORT = '8000';

    // 自动构造前后端基础 URL（注意：apiBase 要包含 /api，因为后端接口在 /api 下）
    const frontendBase = port
        ? `${protocol}//${hostname}:${port}`
        : `${protocol}//${hostname}`;
    const apiBase = `${protocol}//${hostname}:8000/api`;  // 包含 /api 后缀

    CONFIG._frontendBase = frontendBase;
    CONFIG._apiBase = apiBase;  // 修正：包含 /api
    CONFIG._wsBase = `${protocol.replace('http', 'ws')}//${hostname}:8000/ws/stream`;

    console.log(`✅ 自动检测到部署地址: ${frontendBase}，后端 API: ${apiBase}`);
}

// 页面加载时立即自动检测（同步，在任何请求之前执行）
autoDetectFromBrowser();

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

    // 3. 回退：从浏览器 URL 动态构造（前后端同服务器）
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000/api`;
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

    // 3. 回退：从浏览器 URL 动态构造（前后端同服务器）
    const protocol = window.location.protocol.replace('http', 'ws');
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000/ws/stream`;
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
        // 用浏览器地址构建基础 URL，始终以此为基准
        const apiBase = CONFIG._apiBase || `${location.protocol}//${location.hostname}:8000/api`;
        const response = await fetch(`${apiBase}/server_info`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const serverInfo = await response.json();

        // 只同步非地址字段，地址信息永远以浏览器 URL 为准
        // 避免后端返回的内网 IP（如 172.x、192.x）污染前端配置
        if (serverInfo.frontend_port) CONFIG.FRONTEND_PORT = serverInfo.frontend_port;

        // 标记初始化完成（地址已在 autoDetectFromBrowser 中设置，无需覆盖）
        CONFIG._initialized = true;

        console.log('✅ 服务器配置同步成功:', {
            api_base: CONFIG._apiBase,
            ws_base: CONFIG._wsBase,
            frontend_base: CONFIG._frontendBase
        });

        return true;
    } catch (error) {
        console.warn('⚠️ 无法获取服务器配置，使用自动检测配置:', error.message);
        return false;
    }
}

// ==================== 配置加载/保存 ====================

async function loadConfig() {
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        const response = await fetch(`${getApiUrl()}/config`, { headers });
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
        const token = localStorage.getItem('access_token');
        const headers = { 'Content-Type': 'application/json' };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        const response = await fetch(`${getApiUrl()}/config`, {
            method: 'POST',
            headers: headers,
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
