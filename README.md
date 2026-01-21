# 甘肃汇森信息科技有限公司 - 官方网站系统

## 📁 项目结构

```
huisen/
├── index.html          # 前端页面（主页）
├── init.sql            # 数据库初始化脚本
├── api/                # 后端 API 文件夹
│   └── api.php        # API 接口文件
├── config/             # 配置文件文件夹
│   └── config.php     # 数据库配置文件
└── README.md          # 项目说明文档
```

## 🚀 部署步骤

### 1. 数据库配置

1. 打开 **phpStudy** 面板，确保 MySQL 服务已启动
2. 创建数据库 `huisen`（用户名：`huisen`，密码：`123456`）
3. 在 phpMyAdmin 中导入 `init.sql` 文件

### 2. 修改数据库配置

编辑 `config/config.php` 文件，根据实际情况修改以下参数：

```php
define('DB_HOST', 'localhost');        // 数据库主机
define('DB_PORT', '3306');             // 数据库端口
define('DB_NAME', 'huisen');           // 数据库名
define('DB_USER', 'huisen');           // 数据库用户名
define('DB_PASS', '123456');           // 数据库密码
```

### 3. 访问网站

- 本地访问：`http://localhost/huisen/`
- 或配置的域名访问

## 📡 API 接口说明

所有 API 接口位于 `api/api.php`：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/api.php?action=stats` | GET | 获取所有业务统计数据 |
| `/api/api.php?action=summary` | GET | 获取汇总数据 |
| `/api/api.php?action=channels` | GET | 获取渠道列表 |
| `/api/api.php?action=chart_data` | GET | 获取图表数据 |
| `/api/api.php?action=add` | POST | 添加新记录 |
| `/api/api.php?action=init_test` | GET | 初始化测试数据 |

## 🤖 Coze AI 智能体

- **Bot ID**: `7595849107479543808`
- **Token**: 已配置在 `index.html` 中
- **功能**: 点击"启动 AI 对话"按钮或导航栏"AI 助手"即可打开聊天窗口

## 🔧 常见问题

### 1. API 接口无法访问
- 检查 `config/config.php` 中的数据库配置是否正确
- 确保 MySQL 服务已启动
- 检查文件路径是否正确

### 2. Coze 智能体无法打开
- 检查网络连接（SDK 需要从 CDN 加载）
- 打开浏览器控制台查看错误信息
- 确认 Token 是否有效

### 3. 数据无法显示
- 访问 `/api/api.php?action=init_test` 初始化测试数据
- 检查数据库连接是否正常

## 📝 更新日志

### 2024-01-19
- ✅ 重构文件结构，将 PHP 文件分类管理
- ✅ 修复 Coze 智能体启动对话功能
- ✅ 优化 API 路径引用
- ✅ 改进错误处理和调试信息
