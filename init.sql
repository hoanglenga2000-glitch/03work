-- ============================================
-- 甘肃汇森信息科技有限公司 - 数据库初始化脚本
-- 运行环境: MySQL 5.7+
-- 字符集: utf8mb4
-- ============================================

-- 如果数据库不存在则创建（使用phpMyAdmin或命令行执行）
-- CREATE DATABASE IF NOT EXISTS `huisen` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 选择数据库
-- USE `huisen`;

-- ============================================
-- 业务统计数据表 (business_stats)
-- ============================================
DROP TABLE IF EXISTS `business_stats`;

CREATE TABLE `business_stats` (
    `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `channel_name` VARCHAR(100) NOT NULL COMMENT '渠道名称',
    `new_adds` INT(11) DEFAULT 0 COMMENT '新增',
    `broadband` INT(11) DEFAULT 0 COMMENT '宽带',
    `new_coins` INT(11) DEFAULT 0 COMMENT '新增金币',
    `stock_coins` INT(11) DEFAULT 0 COMMENT '存量金币',
    `low_commission` INT(11) DEFAULT 0 COMMENT '低提',
    `gigabit` INT(11) DEFAULT 0 COMMENT '千兆',
    `family_net` INT(11) DEFAULT 0 COMMENT '亲情网',
    `mobile_home` INT(11) DEFAULT 0 COMMENT '移动爱家',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    INDEX `idx_channel_name` (`channel_name`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务统计数据表';

-- ============================================
-- 插入测试数据（基于截图中的数据）
-- ============================================
INSERT INTO `business_stats` 
(`channel_name`, `new_adds`, `broadband`, `new_coins`, `stock_coins`, `low_commission`, `gigabit`, `family_net`, `mobile_home`) 
VALUES 
('七里河恒巨', 85, 34, 44, 0, 405, 38, 9, 0),
('城关汇达旗舰店', 81, 27, 46, 0, 0, 0, 5, 0),
('西固冯立超', 90, 46, 39, 73, 0, 0, 5, 0),
('西固金恒生', 5, 0, 0, 3, 0, 0, 1, 0),
('城关恒巨', 85, 37, 22, 0, 0, 0, 5, 0),
('汇森同创', 0, 0, 0, 70, 1, 0, 0, 0),
('西峰区统办楼', 0, 0, 0, 57, 0, 0, 0, 0),
('安定区物美超市', 0, 0, 0, 115, 0, 0, 0, 0),
('成县于军旗', 0, 0, 0, 45, 0, 0, 0, 0);

-- ============================================
-- 创建用户信息表（可选，用于后续扩展）
-- ============================================
DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
    `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户ID',
    `username` VARCHAR(50) NOT NULL COMMENT '用户名',
    `password` VARCHAR(255) NOT NULL COMMENT '密码(加密)',
    `real_name` VARCHAR(50) DEFAULT NULL COMMENT '真实姓名',
    `role` ENUM('admin', 'manager', 'staff') DEFAULT 'staff' COMMENT '角色',
    `status` TINYINT(1) DEFAULT 1 COMMENT '状态：1正常 0禁用',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户信息表';

-- 插入默认管理员账号（密码: admin123，实际使用请修改）
INSERT INTO `users` (`username`, `password`, `real_name`, `role`) 
VALUES ('admin', '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', '系统管理员', 'admin');

-- ============================================
-- 查询验证
-- ============================================
-- SELECT * FROM business_stats;
-- SELECT * FROM users;
