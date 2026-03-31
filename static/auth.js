/**
 * 认证状态管理模块
 * 处理用户登录状态、Token 管理、路由守卫等
 */

// ============ 配置 ============
const API_BASE = localStorage.getItem('api_base') || `${window.location.protocol}//${window.location.hostname}:8000/api`;

// ============ 认证状态 ============
let currentUser = null;
let authStateCallbacks = [];

/**
 * 获取当前用户信息
 */
function getCurrentUser() {
    if (currentUser) return currentUser;
    
    const userStr = localStorage.getItem('user');
    if (userStr) {
        try {
            currentUser = JSON.parse(userStr);
            return currentUser;
        } catch (e) {
            console.error('解析用户信息失败:', e);
            return null;
        }
    }
    return null;
}

/**
 * 检查是否已登录
 */
function isLoggedIn() {
    const token = localStorage.getItem('access_token');
    return !!token;
}

/**
 * 获取访问令牌
 */
function getAccessToken() {
    return localStorage.getItem('access_token');
}

/**
 * 获取刷新令牌
 */
function getRefreshToken() {
    return localStorage.getItem('refresh_token');
}

/**
 * 设置用户登录状态
 */
function setLogin(user, accessToken, refreshToken) {
    currentUser = user;
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
    localStorage.setItem('user', JSON.stringify(user));
    
    // 触发状态变更回调
    notifyAuthStateChange(true, user);
}

/**
 * 清除登录状态
 */
function logout() {
    currentUser = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    localStorage.removeItem('is_guest');   // ← 补上这行
    
    notifyAuthStateChange(false, null);
}

/**
 * 刷新访问令牌
 */
async function refreshAccessToken() {
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/auth/refresh`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
            logout();
            return false;
        }

        const data = await response.json();
        if (data.success) {
            localStorage.setItem('access_token', data.access_token);
            return true;
        }
        
        logout();
        return false;
    } catch (error) {
        console.error('刷新 Token 失败:', error);
        logout();
        return false;
    }
}

/**
 * 获取当前用户信息（从服务器）
 */
async function fetchCurrentUser() {
    const token = getAccessToken();
    if (!token) {
        return null;
    }

    try {
        const response = await fetch(`${API_BASE}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            if (response.status === 401) {
                // 尝试刷新 Token
                const refreshed = await refreshAccessToken();
                if (refreshed) {
                    return fetchCurrentUser();
                }
            }
            return null;
        }

        const user = await response.json();
        localStorage.setItem('user', JSON.stringify(user));
        currentUser = user;
        return user;
    } catch (error) {
        console.error('获取用户信息失败:', error);
        return null;
    }
}

/**
 * 添加认证状态变更监听器
 */
function addAuthStateListener(callback) {
    authStateCallbacks.push(callback);
}

/**
 * 移除认证状态变更监听器
 */
function removeAuthStateListener(callback) {
    authStateCallbacks = authStateCallbacks.filter(cb => cb !== callback);
}

/**
 * 通知认证状态变更
 */
function notifyAuthStateChange(isLoggedIn, user) {
    authStateCallbacks.forEach(callback => {
        try {
            callback(isLoggedIn, user);
        } catch (e) {
            console.error('Auth state callback error:', e);
        }
    });
}

// ============ 多租户辅助函数 ============

/**
 * 获取当前用户的 tenant_uuid
 */
function getTenantUuid() {
    const user = getCurrentUser();
    if (user && user.tenant_uuid !== undefined && user.tenant_uuid !== null) {
        return user.tenant_uuid;
    }
    return null;
}

/**
 * 🆕 获取当前用户的 tenant_id（兼容旧接口）
 */
function getTenantId() {
    return getTenantUuid();
}

/**
 * 将 tenant_uuid 添加到 URL 参数中（用于后端从请求头 Authorization 解析）
 * 方式：通过 Authorization header，后端从 JWT payload 中读取 tenant_uuid
 */
function appendTenantUuidToUrl(url) {
    const tenantUuid = getTenantUuid();
    if (tenantUuid === null) {
        return url;
    }
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}_tenant_uuid=${tenantUuid}`;
}


// ============ 请求拦截器 ============
/**
 * 带认证的请求方法
 * 自动携带 Authorization header（JWT token）和 tenant_id
 */
async function authFetch(url, options = {}) {
    const token = getAccessToken();
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    if (token) {
        defaultOptions.headers['Authorization'] = `Bearer ${token}`;
    }

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    let response = await fetch(url, mergedOptions);

    // 如果收到 401，尝试刷新 Token
    if (response.status === 401 && token) {
        const refreshed = await refreshAccessToken();
        if (refreshed) {
            // 重新设置 Authorization header
            mergedOptions.headers['Authorization'] = `Bearer ${getAccessToken()}`;
            response = await fetch(url, mergedOptions);
        } else {
            // 刷新失败，跳转到登录页
            window.location.href = '/login.html';
            return null;
        }
    }

    return response;
}

// ============ 路由守卫 ============
/**
 * 检查是否需要登录
 * 如果未登录，重定向到登录页
 */
function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = '/login.html';
        return false;
    }
    return true;
}

/**
 * 如果已登录，重定向到首页
 */
function redirectIfLoggedIn() {
    if (isLoggedIn()) {
        window.location.href = '/index.html';
        return true;
    }
    return false;
}

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    // 页面加载时检查登录状态
    if (isLoggedIn()) {
        fetchCurrentUser().then(user => {
            if (!user) {
                // Token 已失效
                window.location.href = '/login.html';
            }
        });
    }

    // 监听页面可见性变化，刷新用户状态
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && isLoggedIn()) {
            fetchCurrentUser();
        }
    });

    console.log('✅ auth.js 已加载');
});

// ============ 导出到全局 ============
window.auth = {
    getCurrentUser,
    isLoggedIn,
    getAccessToken,
    getRefreshToken,
    setLogin,
    logout,
    refreshAccessToken,
    fetchCurrentUser,
    addAuthStateListener,
    removeAuthStateListener,
    notifyAuthStateChange,
    requireAuth,
    redirectIfLoggedIn,
    authFetch,
    getTenantId,     // 🆕
    getTenantUuid,    // 🆕 新增
    appendTenantUuidToUrl,  // 🆕 新增
    API_BASE,
};
