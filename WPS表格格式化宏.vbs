' WPS 表格自动调整 VBScript 脚本
' 功能：根据窗口自动调整所有表格宽度
' 使用方法：双击运行此文件，会自动调整当前打开的 WPS 文档中的所有表格

On Error Resume Next

' 连接到 WPS
Set wpsApp = CreateObject("kwps.Application")

If Err.Number <> 0 Then
    MsgBox "错误：无法连接到 WPS" & vbCrLf & "请确保 WPS 已打开并且有文档打开", vbCritical, "错误"
    WScript.Quit
End If

' 检查是否有打开的文档
If wpsApp.Documents.Count = 0 Then
    MsgBox "错误：没有打开的 WPS 文档" & vbCrLf & "请先打开要处理的文档", vbCritical, "错误"
    WScript.Quit
End If

' 获取活动文档
Set doc = wpsApp.ActiveDocument
docName = doc.Name

' 检查是否有表格
If doc.Tables.Count = 0 Then
    MsgBox "警告：文档中没有找到表格", vbExclamation, "提示"
    WScript.Quit
End If

' 开始处理
tablesCount = doc.Tables.Count
processedCount = 0

For i = 1 To tablesCount
    On Error Resume Next
    Set table = doc.Tables(i)
    
    If Err.Number = 0 Then
        ' 根据窗口自动调整表格宽度
        On Error Resume Next
        table.AutoFitBehavior 1  ' 按窗口调整 (wdAutoFitWindow = 1)
        If Err.Number <> 0 Then
            Err.Clear
            ' 如果按窗口调整失败，尝试按内容调整
            table.AutoFitBehavior 0  ' 按内容调整 (wdAutoFitContent = 0)
        End If
        
        If Err.Number = 0 Then
            processedCount = processedCount + 1
        End If
    End If
    Err.Clear
Next

' 显示结果
MsgBox "完成！" & vbCrLf & "文档：" & docName & vbCrLf & "已调整表格数：" & processedCount & "/" & tablesCount, vbInformation, "表格调整完成"

' 清理
Set doc = Nothing
Set wpsApp = Nothing
