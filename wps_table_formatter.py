"""
WPS 文档表格自动调整脚本
功能：所有表格根据窗口自动调整宽度
"""

import win32com.client
import sys

def format_wps_tables():
    """自动调整 WPS 文档中所有表格的宽度"""
    try:
        # 连接到 WPS 应用程序
        print("正在连接到 WPS...")
        wps_app = win32com.client.Dispatch("kwps.Application")
        
        if wps_app.Documents.Count == 0:
            print("[错误] 没有打开的 WPS 文档")
            print("请先打开要处理的 WPS 文档")
            return False
        
        # 获取当前活动文档
        doc = wps_app.ActiveDocument
        print(f"[成功] 已连接到文档: {doc.Name}")
        
        # 获取文档中的所有表格
        tables_count = doc.Tables.Count
        if tables_count == 0:
            print("[警告] 文档中没有找到表格")
            return False
        
        print(f"[信息] 找到 {tables_count} 个表格，开始自动调整...")
        
        # 遍历所有表格
        success_count = 0
        for i in range(1, tables_count + 1):
            try:
                table = doc.Tables(i)
                print(f"处理表格 {i}/{tables_count}...", end=" ")
                
                # 根据窗口自动调整表格宽度
                try:
                    # 方法1: 根据窗口调整 (wdAutoFitWindow = 1)
                    table.AutoFitBehavior(1)
                    print("[完成]")
                    success_count += 1
                except:
                    # 方法2: 如果方法1失败，尝试根据内容调整 (wdAutoFitContent = 0)
                    try:
                        table.AutoFitBehavior(0)
                        print("[完成 - 使用备用方法]")
                        success_count += 1
                    except Exception as e:
                        print(f"[失败] {str(e)}")
                        continue
                
            except Exception as e:
                print(f"[错误] 处理表格 {i} 时出错: {str(e)}")
                continue
        
        print(f"\n[完成] 成功处理 {success_count}/{tables_count} 个表格")
        print("[提示] 请记得保存文档 (Ctrl+S)")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "kwps.Application" in error_msg or "CreateObject" in error_msg:
            print("[错误] 无法连接到 WPS")
            print("请确保：")
            print("  1. WPS 已安装并正在运行")
            print("  2. 至少有一个 WPS 文档已打开")
            print("  3. WPS 支持 COM 接口（WPS Office 专业版或企业版）")
        else:
            print(f"[错误] 发生错误: {error_msg}")
        return False

if __name__ == "__main__":
        print("=" * 50)
        print("WPS 表格自动调整工具")
        print("功能：根据窗口自动调整所有表格宽度")
        print("=" * 50)
        print()
        
        # 设置控制台编码为 UTF-8（如果可能）
        try:
            import sys
            import io
            if sys.platform == 'win32':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass
        
        # 检查是否安装了 pywin32
        try:
            import win32com.client
        except ImportError:
            print("[错误] 未安装 pywin32 库")
            print("请运行以下命令安装：")
            print("  pip install pywin32")
            sys.exit(1)
        
        # 运行表格自动调整
        success = format_wps_tables()
        
        if success:
            print("\n[完成] 脚本执行完成")
        else:
            print("\n[失败] 脚本执行失败")
            input("\n按 Enter 键退出...")
