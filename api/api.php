<?php
/**
 * ============================================
 * 甘肃汇森信息科技有限公司 - API 接口
 * ============================================
 * 
 * 接口说明：
 * GET  /api/api.php?action=stats       获取所有业务统计数据
 * GET  /api/api.php?action=summary     获取汇总数据
 * GET  /api/api.php?action=channels    获取渠道列表
 * POST /api/api.php?action=add         添加新记录
 * GET  /api/api.php?action=init_test   初始化测试数据
 */

// 引入配置文件（使用相对路径）
require_once __DIR__ . '/../config/config.php';

// 获取请求方法和操作类型
$method = $_SERVER['REQUEST_METHOD'];
$action = isset($_GET['action']) ? $_GET['action'] : 'stats';

// 处理OPTIONS预检请求（CORS）
if ($method === 'OPTIONS') {
    header('Access-Control-Allow-Origin: *');
    header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
    header('Access-Control-Allow-Headers: Content-Type');
    exit(0);
}

// 获取数据库实例
$db = Database::getInstance();

// ============================================
// 路由处理
// ============================================
try {
    switch ($action) {
        case 'stats':
            // 获取所有业务统计数据
            getStats($db);
            break;
            
        case 'summary':
            // 获取汇总数据
            getSummary($db);
            break;
            
        case 'channels':
            // 获取渠道列表
            getChannels($db);
            break;
            
        case 'add':
            // 添加新记录
            if ($method !== 'POST') {
                errorResponse('请使用POST方法', 405);
            }
            addRecord($db);
            break;
            
        case 'init_test':
            // 初始化测试数据
            initTestData($db);
            break;
            
        case 'chart_data':
            // 获取图表数据
            getChartData($db);
            break;
            
        default:
            errorResponse('未知的操作类型', 400);
    }
} catch (Exception $e) {
    errorResponse($e->getMessage(), 500);
}

// ============================================
// API 函数定义
// ============================================

/**
 * 获取所有业务统计数据
 */
function getStats($db) {
    $sql = "SELECT * FROM business_stats ORDER BY created_at DESC";
    $data = $db->fetchAll($sql);
    
    // 如果没有数据，返回空数组
    if (empty($data)) {
        successResponse([], '暂无数据，请先初始化测试数据');
        return;
    }
    
    successResponse($data, '获取成功');
}

/**
 * 获取汇总数据（用于Dashboard顶部指标卡片）
 */
function getSummary($db) {
    $sql = "SELECT 
                SUM(new_adds) as total_new_adds,
                SUM(broadband) as total_broadband,
                SUM(new_coins) as total_new_coins,
                SUM(stock_coins) as total_stock_coins,
                SUM(low_commission) as total_low_commission,
                SUM(gigabit) as total_gigabit,
                SUM(family_net) as total_family_net,
                SUM(mobile_home) as total_mobile_home,
                COUNT(*) as channel_count
            FROM business_stats";
    
    $summary = $db->fetchOne($sql);
    
    // 计算总计
    $summary['total_all'] = (int)$summary['total_new_adds'] + 
                            (int)$summary['total_broadband'] + 
                            (int)$summary['total_new_coins'];
    
    // 转换为整数
    foreach ($summary as $key => $value) {
        $summary[$key] = (int)$value;
    }
    
    successResponse($summary, '获取汇总成功');
}

/**
 * 获取渠道列表
 */
function getChannels($db) {
    $sql = "SELECT DISTINCT channel_name FROM business_stats ORDER BY channel_name";
    $data = $db->fetchAll($sql);
    
    $channels = array_column($data, 'channel_name');
    successResponse($channels, '获取渠道列表成功');
}

/**
 * 添加新记录
 */
function addRecord($db) {
    // 获取POST数据
    $input = json_decode(file_get_contents('php://input'), true);
    
    if (empty($input)) {
        $input = $_POST;
    }
    
    // 验证必填字段
    if (empty($input['channel_name'])) {
        errorResponse('渠道名称不能为空');
    }
    
    // 准备插入数据
    $data = [
        'channel_name' => $input['channel_name'],
        'new_adds' => isset($input['new_adds']) ? (int)$input['new_adds'] : 0,
        'broadband' => isset($input['broadband']) ? (int)$input['broadband'] : 0,
        'new_coins' => isset($input['new_coins']) ? (int)$input['new_coins'] : 0,
        'stock_coins' => isset($input['stock_coins']) ? (int)$input['stock_coins'] : 0,
        'low_commission' => isset($input['low_commission']) ? (int)$input['low_commission'] : 0,
        'gigabit' => isset($input['gigabit']) ? (int)$input['gigabit'] : 0,
        'family_net' => isset($input['family_net']) ? (int)$input['family_net'] : 0,
        'mobile_home' => isset($input['mobile_home']) ? (int)$input['mobile_home'] : 0
    ];
    
    $id = $db->insert('business_stats', $data);
    
    successResponse(['id' => $id], '添加成功');
}

/**
 * 初始化测试数据
 */
function initTestData($db) {
    // 检查是否已有数据
    $count = $db->fetchOne("SELECT COUNT(*) as cnt FROM business_stats");
    
    if ($count['cnt'] > 0) {
        successResponse(['exists' => true], '数据已存在，无需重复初始化');
        return;
    }
    
    // 插入测试数据（基于截图）
    $testData = [
        ['七里河恒巨', 85, 34, 44, 0, 405, 38, 9, 0],
        ['城关汇达旗舰店', 81, 27, 46, 0, 0, 0, 5, 0],
        ['西固冯立超', 90, 46, 39, 73, 0, 0, 5, 0],
        ['西固金恒生', 5, 0, 0, 3, 0, 0, 1, 0],
        ['城关恒巨', 85, 37, 22, 0, 0, 0, 5, 0],
        ['汇森同创', 0, 0, 0, 70, 1, 0, 0, 0],
        ['西峰区统办楼', 0, 0, 0, 57, 0, 0, 0, 0],
        ['安定区物美超市', 0, 0, 0, 115, 0, 0, 0, 0],
        ['成县于军旗', 0, 0, 0, 45, 0, 0, 0, 0]
    ];
    
    $sql = "INSERT INTO business_stats 
            (channel_name, new_adds, broadband, new_coins, stock_coins, low_commission, gigabit, family_net, mobile_home) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
    
    foreach ($testData as $row) {
        $db->query($sql, $row);
    }
    
    successResponse(['inserted' => count($testData)], '测试数据初始化成功');
}

/**
 * 获取图表数据
 */
function getChartData($db) {
    // 获取各渠道数据用于图表
    $sql = "SELECT channel_name, new_adds, broadband, new_coins, stock_coins, 
                   low_commission, gigabit, family_net, mobile_home
            FROM business_stats 
            ORDER BY new_adds DESC";
    
    $data = $db->fetchAll($sql);
    
    // 准备图表数据格式
    $chartData = [
        'channels' => array_column($data, 'channel_name'),
        'newAdds' => array_map('intval', array_column($data, 'new_adds')),
        'broadband' => array_map('intval', array_column($data, 'broadband')),
        'newCoins' => array_map('intval', array_column($data, 'new_coins')),
        'stockCoins' => array_map('intval', array_column($data, 'stock_coins')),
        'lowCommission' => array_map('intval', array_column($data, 'low_commission')),
        'gigabit' => array_map('intval', array_column($data, 'gigabit')),
        'familyNet' => array_map('intval', array_column($data, 'family_net')),
        'mobileHome' => array_map('intval', array_column($data, 'mobile_home')),
        'raw' => $data
    ];
    
    successResponse($chartData, '获取图表数据成功');
}
