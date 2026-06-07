/**
 * DeepAgentForce — lightweight EN/ZH i18n
 *
 * Usage:
 *   data-i18n="key"             → sets element.textContent
 *   data-i18n-placeholder="key" → sets element.placeholder
 *   data-i18n-title="key"       → sets element.title
 *   data-i18n-html="key"        → sets element.innerHTML (use sparingly)
 *
 * For index.html the larger page also uses SELECTORS (below) so we do not
 * have to add data-i18n to every element in the 10k-line file.
 */

(function () {
  'use strict';

  // ─── Translations ───────────────────────────────────────────────────────────

  var T = {
    zh: {
      // ── Common ──────────────────────────────────────────────────────────────
      refresh: '刷新',
      save: '保存',
      cancel: '取消',
      delete: '删除',
      close: '关闭',
      confirm: '确认',
      all: '全部',
      add: '添加',
      edit: '编辑',
      test: '测试',

      // ── Sidebar nav ─────────────────────────────────────────────────────────
      'nav.main': '主功能',
      'nav.chat': '智能对话',
      'nav.knowledge': '知识库管理',
      'nav.officeTools': '办公助手',
      'nav.proofread': '文本校对',
      'nav.sessions': '会话',
      'nav.newChat': '新会话',
      'nav.emptySessions': '暂无历史会话',
      'nav.management': '管理工具',
      'nav.skills': '技能管理',
      'nav.config': '模型配置',

      // ── User dropdown ────────────────────────────────────────────────────────
      'user.settings': '个人设置',
      'user.logout': '退出登录',
      'user.member': '成员',

      // ── Header ───────────────────────────────────────────────────────────────
      'header.manual': '手册',
      'header.downloads': '下载',
      'header.connecting': '连接中...',
      'header.connected': '已连接',
      'header.disconnected': '已断开',
      'header.page.chat': '智能对话',
      'header.page.knowledge': '知识库管理',
      'header.page.proofread': '文本校对',
      'header.page.skills': '技能管理',
      'header.page.config': '模型配置',

      // ── Chat ─────────────────────────────────────────────────────────────────
      'chat.welcome': '你好，我是 DeepAgentForce',
      'chat.welcomeSub': '读文档、查知识、用工具、做分析 —— 试试下面的例子，或直接开始提问。',
      'chat.placeholder': '输入消息...',

      // ── Knowledge base ────────────────────────────────────────────────────────
      'kb.title': '知识库管理',
      'kb.subtitle': '管理和查询你的文档库，支持 PDF、Word、TXT 等多种格式',
      'kb.upload': '上传文档',
      'kb.dropText': '拖拽文件到此处或点击上传',
      'kb.dropHint': '支持批量上传多个文件',
      'kb.searchPlaceholder': '搜索文档名称...',
      'kb.selectAll': '全选',
      'kb.batchDelete': '批量删除',
      'kb.filterAll': '全部',
      'kb.filterText': '文本',
      'kb.sortRecent': '最近更新',
      'kb.sortOldest': '最早上传',
      'kb.noSelection': '未选择文档',
      'kb.docCount': '文档数量',
      'kb.chunkCount': '文本块数',
      'kb.totalSize': '总大小',
      'kb.fileDistribution': '文件分布',
      'kb.noCategory': '暂无分类',
      'kb.emptyTitle': '暂无文档',
      'kb.emptyText': '上传文档后即可开始使用知识库进行智能问答',
      'kb.selectDoc': '选择一个文档',
      'kb.selectDocHint': '左侧列表会显示你的全部知识库文档。点击任意一项，右侧会展开详情、元数据和文本预览。',
      'kb.contentPreview': '内容预览',

      // ── Skills page ───────────────────────────────────────────────────────────
      'skills.title': '技能管理',
      'skills.subtitle': '管理智能体的技能插件，扩展系统能力',
      'skills.addSkill': '添加技能',
      'skills.addTool': '添加工具',
      'skills.addMcp': '添加 MCP Server',
      'skills.tabSkills': '技能',
      'skills.tabTools': '工具',
      'skills.tabMcp': 'MCP',
      'skills.installed': '已安装技能',
      'skills.scripts': '脚本总数',
      'skills.builtin': '内置技能',
      'skills.custom': '自定义技能',
      'skills.searchPlaceholder': '搜索技能名称、描述、作者或标签...',
      'skills.sortUpdated': '最近更新',
      'skills.sortName': '按名称',
      'skills.sortScripts': '脚本数量',
      'skills.filterAll': '全部',
      'skills.filterBuiltin': '内置',
      'skills.filterCustom': '自定义',
      'skills.listTitle': '技能列表',
      // 工具 / MCP 子页（区块标题 + 统计标签）
      'tools.customToolsTitle': '自定义工具',
      'tools.readonlyTitle': '内置 & MCP 工具',
      'tools.builtinTools': '内置工具',
      'tools.mcpTools': 'MCP 工具',
      'tools.customTools': '自定义工具',
      'mcp.serverTitle': 'MCP Server',
      'common.readonly': '只读',
      'skills.listHint': '点击任意技能查看详情',
      'skills.emptyTitle': '暂无匹配的技能',
      'skills.emptyText': '试试清空搜索或切换筛选条件，或者创建一个新的自定义技能。',

      // ── Config page ───────────────────────────────────────────────────────────
      'config.title': '模型配置',

      // ── Login ─────────────────────────────────────────────────────────────────
      'login.pageTitle': '登录 - DeepAgentForce',
      'login.heroTitle': '智能研究助手',
      'login.heroDesc': '由 AI 驱动的智能研究平台，帮助您加速洞察、提升效率、做出更好的决策。',
      'login.featureAI': 'AI Agent',
      'login.featureKB': '知识库',
      'login.featureRAG': 'RAG 检索',
      'login.systemOk': '系统运行正常',
      'login.encrypted': '数据加密中',
      'login.welcome': '欢迎回来 👋',
      'login.welcomeDesc': '登录到您的 DeepAgentForce 账户',
      'login.usernameLabel': '用户名 / 邮箱',
      'login.usernamePlaceholder': '请输入用户名或邮箱',
      'login.passwordLabel': '密码',
      'login.passwordPlaceholder': '请输入密码',
      'login.loginBtn': '登 录',
      'login.loggingIn': '登录中...',
      'login.or': '或',
      'login.noAccount': '还没有账号？',
      'login.registerLink': '立即注册',
      'login.validationFill': '请填写用户名和密码',
      'login.loginFailed': '登录失败，请检查账号和密码',
      'login.loginSuccess': '登录成功！正在跳转...',
      'login.loginError': '登录失败',

      // ── Register ──────────────────────────────────────────────────────────────
      'register.pageTitle': '注册 - DeepAgentForce',
      'register.title': '创建账号',
      'register.subtitle': '开启您的智能研究之旅',
      'register.usernameLabel': '用户名',
      'register.usernamePlaceholder': '3-20个字符，可使用字母、数字、下划线',
      'register.emailLabel': '邮箱',
      'register.emailPlaceholder': '用于找回密码和接收通知',
      'register.passwordLabel': '密码',
      'register.passwordPlaceholder': '至少6个字符',
      'register.confirmLabel': '确认密码',
      'register.confirmPlaceholder': '再次输入密码',
      'register.submitBtn': '创建账号',
      'register.or': '或',
      'register.hasAccount': '已有账号？',
      'register.loginLink': '立即登录',
    },

    en: {
      // ── Common ──────────────────────────────────────────────────────────────
      refresh: 'Refresh',
      save: 'Save',
      cancel: 'Cancel',
      delete: 'Delete',
      close: 'Close',
      confirm: 'Confirm',
      all: 'All',
      add: 'Add',
      edit: 'Edit',
      test: 'Test',

      // ── Sidebar nav ─────────────────────────────────────────────────────────
      'nav.main': 'Main',
      'nav.chat': 'Chat',
      'nav.knowledge': 'Knowledge Base',
      'nav.officeTools': 'Office Tools',
      'nav.proofread': 'Proofreading',
      'nav.sessions': 'Sessions',
      'nav.newChat': 'New Chat',
      'nav.emptySessions': 'No conversation history',
      'nav.management': 'Management',
      'nav.skills': 'Skills',
      'nav.config': 'Settings',

      // ── User dropdown ────────────────────────────────────────────────────────
      'user.settings': 'Settings',
      'user.logout': 'Sign Out',
      'user.member': 'Member',

      // ── Header ───────────────────────────────────────────────────────────────
      'header.manual': 'Docs',
      'header.downloads': 'Downloads',
      'header.connecting': 'Connecting...',
      'header.connected': 'Connected',
      'header.disconnected': 'Disconnected',
      'header.page.chat': 'Chat',
      'header.page.knowledge': 'Knowledge Base',
      'header.page.proofread': 'Proofreading',
      'header.page.skills': 'Skills',
      'header.page.config': 'Settings',

      // ── Chat ─────────────────────────────────────────────────────────────────
      'chat.welcome': "Hi, I'm DeepAgentForce",
      'chat.welcomeSub': 'Read documents, search knowledge, use tools, analyze data — try an example below, or just ask.',
      'chat.placeholder': 'Type a message...',

      // ── Knowledge base ────────────────────────────────────────────────────────
      'kb.title': 'Knowledge Base',
      'kb.subtitle': 'Manage your document library. Supports PDF, Word, TXT and more.',
      'kb.upload': 'Upload',
      'kb.dropText': 'Drop files here or click to upload',
      'kb.dropHint': 'Supports batch upload of multiple files',
      'kb.searchPlaceholder': 'Search by document name...',
      'kb.selectAll': 'Select All',
      'kb.batchDelete': 'Delete Selected',
      'kb.filterAll': 'All',
      'kb.filterText': 'Text',
      'kb.sortRecent': 'Recently Updated',
      'kb.sortOldest': 'Oldest First',
      'kb.noSelection': 'No document selected',
      'kb.docCount': 'Documents',
      'kb.chunkCount': 'Chunks',
      'kb.totalSize': 'Total Size',
      'kb.fileDistribution': 'File Types',
      'kb.noCategory': 'No documents yet',
      'kb.emptyTitle': 'No documents',
      'kb.emptyText': 'Upload documents to start using the knowledge base for intelligent Q&A',
      'kb.selectDoc': 'Select a document',
      'kb.selectDocHint': 'The list shows all your knowledge base documents. Click any item to see details, metadata and a text preview.',
      'kb.contentPreview': 'Content Preview',

      // ── Skills page ───────────────────────────────────────────────────────────
      'skills.title': 'Skills',
      'skills.subtitle': 'Manage agent skill plugins and extend system capabilities',
      'skills.addSkill': 'Add Skill',
      'skills.addTool': 'Add Tool',
      'skills.addMcp': 'Add MCP Server',
      'skills.tabSkills': 'Skills',
      'skills.tabTools': 'Tools',
      'skills.tabMcp': 'MCP',
      'skills.installed': 'Installed',
      'skills.scripts': 'Total Scripts',
      'skills.builtin': 'Built-in',
      'skills.custom': 'Custom',
      'skills.searchPlaceholder': 'Search by name, description, author or tag...',
      'skills.sortUpdated': 'Recently Updated',
      'skills.sortName': 'By Name',
      'skills.sortScripts': 'Script Count',
      'skills.filterAll': 'All',
      'skills.filterBuiltin': 'Built-in',
      'skills.filterCustom': 'Custom',
      'skills.listTitle': 'Skill List',
      // Tools / MCP subpanels (section titles + stat labels)
      'tools.customToolsTitle': 'Custom Tools',
      'tools.readonlyTitle': 'Built-in & MCP Tools',
      'tools.builtinTools': 'Built-in Tools',
      'tools.mcpTools': 'MCP Tools',
      'tools.customTools': 'Custom Tools',
      'mcp.serverTitle': 'MCP Server',
      'common.readonly': 'Read-only',
      'skills.listHint': 'Click any skill to view details',
      'skills.emptyTitle': 'No matching skills',
      'skills.emptyText': 'Try clearing the search or switching filters, or create a new custom skill.',

      // ── Config page ───────────────────────────────────────────────────────────
      'config.title': 'Settings',

      // ── Login ─────────────────────────────────────────────────────────────────
      'login.pageTitle': 'Sign In - DeepAgentForce',
      'login.heroTitle': 'AI Research Assistant',
      'login.heroDesc': 'An AI-powered research platform that accelerates insights, boosts productivity, and helps you make better decisions.',
      'login.featureAI': 'AI Agent',
      'login.featureKB': 'Knowledge Base',
      'login.featureRAG': 'RAG Search',
      'login.systemOk': 'System Online',
      'login.encrypted': 'Data Encrypted',
      'login.welcome': 'Welcome back 👋',
      'login.welcomeDesc': 'Sign in to your DeepAgentForce account',
      'login.usernameLabel': 'Username / Email',
      'login.usernamePlaceholder': 'Enter username or email',
      'login.passwordLabel': 'Password',
      'login.passwordPlaceholder': 'Enter password',
      'login.loginBtn': 'Sign In',
      'login.loggingIn': 'Signing in...',
      'login.or': 'or',
      'login.noAccount': "Don't have an account?",
      'login.registerLink': 'Register',
      'login.validationFill': 'Please enter username and password',
      'login.loginFailed': 'Login failed. Please check your credentials',
      'login.loginSuccess': 'Login successful! Redirecting...',
      'login.loginError': 'Login failed',

      // ── Register ──────────────────────────────────────────────────────────────
      'register.pageTitle': 'Register - DeepAgentForce',
      'register.title': 'Create Account',
      'register.subtitle': 'Start your AI research journey',
      'register.usernameLabel': 'Username',
      'register.usernamePlaceholder': '3-20 characters, letters, numbers, underscores',
      'register.emailLabel': 'Email',
      'register.emailPlaceholder': 'For password recovery and notifications',
      'register.passwordLabel': 'Password',
      'register.passwordPlaceholder': 'At least 6 characters',
      'register.confirmLabel': 'Confirm Password',
      'register.confirmPlaceholder': 'Re-enter password',
      'register.submitBtn': 'Create Account',
      'register.or': 'or',
      'register.hasAccount': 'Already have an account?',
      'register.loginLink': 'Sign in',
    },
  };

  // ─── Selector map for index.html (avoids mass data-i18n edits) ──────────────
  // Each entry: [cssSelector, translationKey, attribute?]
  // attribute defaults to 'text' (textContent). Use 'placeholder' or 'title' for those.
  var INDEX_SELECTORS = [
    // Sidebar nav section titles
    ['.nav-section-title:nth-of-type(1)', 'nav.main'],
    ['[data-page="chat"] > span', 'nav.chat'],
    ['[data-page="knowledge"] > span', 'nav.knowledge'],
    ['[data-page="proofread"] > span', 'nav.proofread'],
    ['[data-page="skills"] > span', 'nav.skills'],
    ['[data-page="config"] > span', 'nav.config'],
    ['.empty-sessions', 'nav.emptySessions'],
    // User dropdown
    ['.dropdown-item:not(.danger)', 'user.settings'],
    ['.dropdown-item.danger', 'user.logout'],
    // Header buttons
    ['#headerHelpBtn > span', 'header.manual'],
    ['#headerDownloadsBtn > span', 'header.downloads'],
    // Chat welcome
    ['.welcome-text', 'chat.welcome'],
    ['#messageInput', 'chat.placeholder', 'placeholder'],
    // Knowledge page
    ['#knowledgePage .kb-header-text h1', 'kb.title'],
    ['#knowledgePage .kb-header-text p', 'kb.subtitle'],
    // 知识库统计标签改用各自的 data-i18n（nth-of-type 会误把三个标签都当成第 1 个）
    ['#knowledgePage .kb-insight-label', 'kb.fileDistribution'],
    ['.kb-type-pill.muted', 'kb.noCategory'],
    ['.upload-text', 'kb.dropText'],
    ['.upload-hint', 'kb.dropHint'],
    ['#docSearchInput', 'kb.searchPlaceholder', 'placeholder'],
    ['#selectAllBtn', 'kb.selectAll'],
    ['#batchDeleteBtn', 'kb.batchDelete'],
    ['[data-kb-filter="all"]', 'kb.filterAll'],
    ['[data-kb-filter="txt"]', 'kb.filterText'],
    ['#docSortSelect option[value="recent"]', 'kb.sortRecent'],
    ['#docSortSelect option[value="oldest"]', 'kb.sortOldest'],
    ['#emptyState .empty-title', 'kb.emptyTitle'],
    ['#emptyState .empty-text', 'kb.emptyText'],
    ['.kb-drawer-empty-title', 'kb.selectDoc'],
    ['.kb-drawer-empty-text', 'kb.selectDocHint'],
    ['.kb-drawer-section-title', 'kb.contentPreview'],
    // Skills page
    ['.sp-title', 'skills.title'],
    ['#spSubtitle', 'skills.subtitle'],
    ['#skillSearchInput', 'skills.searchPlaceholder', 'placeholder'],
    ['#skillSortSelect option[value="updated"]', 'skills.sortUpdated'],
    ['#skillSortSelect option[value="name"]', 'skills.sortName'],
    ['#skillSortSelect option[value="scripts"]', 'skills.sortScripts'],
    ['[data-skill-filter="all"]', 'skills.filterAll'],
    ['[data-skill-filter="builtin"]', 'skills.filterBuiltin'],
    ['[data-skill-filter="custom"]', 'skills.filterCustom'],
    // ⚠️ 限定到「技能」子页：否则会把「工具/MCP」子页的同名区块标题、统计标签
    //    一并改写成技能文案（工具页/ MCP 页的标题/标签另由各自的 data-i18n 提供）。
    ['.sp-subpanel[data-subpanel="skills"] .sp-section-title', 'skills.listTitle'],
    ['#skillListHint', 'skills.listHint'],
    ['#skillsEmptyState .sp-empty-title', 'skills.emptyTitle'],
    ['#skillsEmptyState .sp-empty-text', 'skills.emptyText'],
    // Stat labels (skills subpanel only)
    ['.sp-subpanel[data-subpanel="skills"] .sp-stat.green .sp-stat-label', 'skills.installed'],
    ['.sp-subpanel[data-subpanel="skills"] .sp-stat.blue .sp-stat-label', 'skills.scripts'],
    ['.sp-subpanel[data-subpanel="skills"] .sp-stat.orange .sp-stat-label', 'skills.builtin'],
    ['.sp-subpanel[data-subpanel="skills"] .sp-stat.purple .sp-stat-label', 'skills.custom'],
    // Subtabs
    ['[data-subtab="skills"]', 'skills.tabSkills'],
    ['[data-subtab="tools"]', 'skills.tabTools'],
    ['[data-subtab="mcp"]', 'skills.tabMcp'],
    // Config
    ['#configPage h1', 'config.title'],
  ];

  // Sidebar section titles are plain-text nodes mixed with SVG — handle separately
  var NAV_SECTION_MAP = [
    ['主功能', 'nav.main'],
    ['会话', 'nav.sessions'],
    ['管理工具', 'nav.management'],
  ];

  // nav-section-title with "办公助手" contains SVG + text node
  var OFFICE_TOOLS_TEXT = '办公助手';

  // ─── Core helpers ────────────────────────────────────────────────────────────

  function t(key) {
    var lang = window.I18n ? window.I18n.lang : (localStorage.getItem('lang') || 'zh');
    return (T[lang] && T[lang][key]) || (T['zh'][key]) || key;
  }

  function setEl(selector, key, attr) {
    var els = document.querySelectorAll(selector);
    if (!els.length) return;
    var val = t(key);
    els.forEach(function (el) {
      if (attr === 'placeholder') {
        el.placeholder = val;
      } else if (attr === 'title') {
        el.title = val;
      } else if (attr === 'html') {
        el.innerHTML = val;
      } else {
        el.textContent = val;
      }
    });
  }

  // Apply data-i18n / data-i18n-placeholder / data-i18n-title attributes
  function applyDataAttrs() {
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(function (el) {
      el.title = t(el.getAttribute('data-i18n-title'));
    });
    document.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });
  }

  // Apply selector-based translations (index.html)
  function applySelectors() {
    INDEX_SELECTORS.forEach(function (item) {
      setEl(item[0], item[1], item[2]);
    });

    // Nav section titles have text mixed with SVG — walk childNodes
    document.querySelectorAll('.nav-section-title').forEach(function (el) {
      for (var i = 0; i < el.childNodes.length; i++) {
        var node = el.childNodes[i];
        if (node.nodeType === 3) { // TEXT_NODE
          var raw = node.textContent.trim();
          NAV_SECTION_MAP.forEach(function (pair) {
            if (raw === pair[0] || raw.indexOf(pair[0]) !== -1) {
              node.textContent = ' ' + t(pair[1]) + ' ';
            }
          });
          // office tools label
          if (raw === OFFICE_TOOLS_TEXT || raw.indexOf(OFFICE_TOOLS_TEXT) !== -1) {
            node.textContent = ' ' + t('nav.officeTools') + ' ';
          }
        }
      }
    });

    // new-chat-btn has SVG + text node
    var newChatBtn = document.querySelector('.new-chat-btn');
    if (newChatBtn) {
      for (var i = 0; i < newChatBtn.childNodes.length; i++) {
        var node = newChatBtn.childNodes[i];
        if (node.nodeType === 3 && node.textContent.trim()) {
          node.textContent = '\n                ' + t('nav.newChat') + '\n            ';
          break;
        }
      }
    }

    // Add Skill / Add Tool / Add MCP Server buttons have SVG + text
    var btnMap = [
      ['#addSkillBtn', 'skills.addSkill'],
      ['.sp-actions-group[data-actions="tools"] .sp-add-btn', 'skills.addTool'],
      ['.sp-actions-group[data-actions="mcp"] .sp-add-btn', 'skills.addMcp'],
    ];
    btnMap.forEach(function (pair) {
      var btn = document.querySelector(pair[0]);
      if (!btn) return;
      for (var i = 0; i < btn.childNodes.length; i++) {
        var node = btn.childNodes[i];
        if (node.nodeType === 3 && node.textContent.trim()) {
          node.textContent = '\n                        ' + t(pair[1]) + '\n                    ';
          break;
        }
      }
    });

    // Refresh buttons (text node after SVG)
    var refreshSelectors = [
      '#refreshKnowledgeBtn',
      '#refreshSkillsBtn',
      '.sp-actions-group[data-actions="tools"] .sp-ghost-btn',
      '.sp-actions-group[data-actions="mcp"] .sp-ghost-btn',
    ];
    refreshSelectors.forEach(function (sel) {
      var btn = document.querySelector(sel);
      if (!btn) return;
      for (var i = 0; i < btn.childNodes.length; i++) {
        var node = btn.childNodes[i];
        if (node.nodeType === 3 && node.textContent.trim()) {
          node.textContent = '\n                    ' + t('refresh') + '\n                ';
          break;
        }
      }
    });

    // Knowledge upload button (has SVG + text)
    var uploadBtn = document.querySelector('.btn-primary-large');
    if (uploadBtn) {
      for (var i = 0; i < uploadBtn.childNodes.length; i++) {
        var node = uploadBtn.childNodes[i];
        if (node.nodeType === 3 && node.textContent.trim()) {
          node.textContent = '\n                                    ' + t('kb.upload') + '\n                                ';
          break;
        }
      }
    }

    // page title
    if (document.title && (document.title.includes('DeepAgentForce') || document.title.includes('智能'))) {
      if (window.I18n && window.I18n.lang === 'en') {
        document.title = 'DeepAgentForce';
      }
    }

    // html lang attribute
    document.documentElement.lang = (window.I18n && window.I18n.lang === 'en') ? 'en' : 'zh-CN';
  }

  // ─── Toggle button ───────────────────────────────────────────────────────────

  var TOGGLE_STYLE = [
    'background:none',
    'border:1px solid var(--border-color,#e5e7eb)',
    'border-radius:6px',
    'color:var(--text-secondary,#475569)',
    'cursor:pointer',
    'font-size:11px',
    'font-weight:600',
    'letter-spacing:.05em',
    'padding:3px 7px',
    'margin-left:auto',
    'flex-shrink:0',
    'transition:background .15s,color .15s',
  ].join(';');

  function injectToggleBtn(containerId) {
    if (document.getElementById('langToggleBtn')) return;
    var container = containerId ? document.getElementById(containerId) : null;
    if (!container) {
      // index.html: put inside .sidebar-header
      container = document.querySelector('.sidebar-header');
    }
    if (!container) return;

    var btn = document.createElement('button');
    btn.id = 'langToggleBtn';
    btn.setAttribute('style', TOGGLE_STYLE);
    btn.setAttribute('title', 'Switch language / 切换语言');
    btn.textContent = (window.I18n && window.I18n.lang === 'zh') ? 'EN' : '中';
    btn.addEventListener('click', function () {
      if (window.I18n) window.I18n.toggle();
    });

    // Make sidebar-header flex so the button sits on the right
    if (container.classList.contains('sidebar-header')) {
      container.style.display = 'flex';
      container.style.alignItems = 'center';
      container.appendChild(btn);
    } else {
      container.appendChild(btn);
    }
  }

  // For login / register pages, inject into .brand or .logo area
  function injectToggleBtnAuth() {
    if (document.getElementById('langToggleBtn')) return;
    var brand = document.querySelector('.brand') || document.querySelector('.logo');
    if (!brand) return;
    var btn = document.createElement('button');
    btn.id = 'langToggleBtn';
    var authStyle = [
      'background:none',
      'border:1px solid rgba(255,255,255,0.15)',
      'border-radius:6px',
      'color:rgba(255,255,255,0.6)',
      'cursor:pointer',
      'font-size:11px',
      'font-weight:600',
      'padding:3px 8px',
      'margin-left:12px',
      'transition:all .15s',
    ].join(';');
    btn.setAttribute('style', authStyle);
    btn.textContent = (window.I18n && window.I18n.lang === 'zh') ? 'EN' : '中';
    btn.addEventListener('click', function () {
      if (window.I18n) window.I18n.toggle();
    });
    brand.appendChild(btn);
  }

  // ─── Public API ──────────────────────────────────────────────────────────────

  window.I18n = {
    lang: localStorage.getItem('lang') || 'zh',

    /** Translate a key directly */
    t: function (key) { return t(key); },

    /** Apply all translations to the current page */
    apply: function () {
      applyDataAttrs();
      applySelectors();
    },

    /** Switch language and re-apply */
    toggle: function () {
      this.lang = this.lang === 'zh' ? 'en' : 'zh';
      localStorage.setItem('lang', this.lang);
      var btn = document.getElementById('langToggleBtn');
      if (btn) btn.textContent = this.lang === 'zh' ? 'EN' : '中';
      this.apply();
    },

    /** Set language explicitly */
    setLang: function (lang) {
      if (lang !== 'zh' && lang !== 'en') return;
      this.lang = lang;
      localStorage.setItem('lang', lang);
      var btn = document.getElementById('langToggleBtn');
      if (btn) btn.textContent = lang === 'zh' ? 'EN' : '中';
      this.apply();
    },

    /** Called by page scripts when dynamic content is added */
    refresh: function () {
      this.apply();
    },

    /** Bootstrap: inject toggle button + apply translations */
    init: function (mode) {
      if (mode === 'auth') {
        injectToggleBtnAuth();
      } else {
        injectToggleBtn();
      }
      this.apply();
    },
  };

  // Auto-init on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      var mode = document.body && document.body.classList.contains('auth-page') ? 'auth' : 'default';
      window.I18n.init(mode);
    });
  } else {
    var mode = document.body && document.body.classList.contains('auth-page') ? 'auth' : 'default';
    window.I18n.init(mode);
  }
})();
