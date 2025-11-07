import os
import pandas as pd
import re
from pathlib import Path
from datetime import datetime

# ==================== 配置区域 ====================
DATASET_DIR = r"E:\LUNAR-THU\datasets\Apache"
TEMPLATES_FILE = os.path.join(DATASET_DIR, "Apache_full.log_templates.csv")
STRUCTURED_FILE = os.path.join(DATASET_DIR, "Apache_full.log_structured.csv")

# 输出文件
OUTPUT_TEMPLATES = os.path.join(DATASET_DIR, "Apache_full.log_templates_fixed.csv")
OUTPUT_STRUCTURED = os.path.join(DATASET_DIR, "Apache_full.log_structured_fixed.csv")

# 任务配置
TASK_START_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = "XiancongMeng"


# ==================== 第一步：修复模板文件 ====================
def fix_template_file(input_file, output_file, backup=True):
    """
    修复 Apache_full.log_templates.csv 中的 E1 模板

    找到: "Access denied with code <*> Error reading POST data"
    替换为: "Access denied with code <*>. Error reading POST data"
    （即把 <*> 改成 <*>. ）
    """
    print(f"\n{'=' * 70}")
    print(f"📝 第一步：修复模板文件")
    print(f"{'=' * 70}")

    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return False, 0

    try:
        # 1. 读取模板文件
        print(f"📖 正在读取模板文件...")
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✓ 读取成功，共 {len(df)} 行模板")

        # 2. 找到 E1 行
        print(f"\n🔍 查找 E1 模板...")
        e1_rows = df[df['EventId'] == 'E1']

        if len(e1_rows) == 0:
            print(f"❌ 找不到 E1 模板")
            return False, 0

        e1_index = e1_rows.index[0]
        original_template = df.loc[e1_index, 'EventTemplate']

        print(f"✓ 找到 E1 模板")
        print(f"\n📋 原始模板:")
        print(f"  {original_template}")

        # 3. 修复模板：只替换 "Access denied with code <*> Error reading POST data" 中的 <*> 为 <*>.
        print(f"\n🔧 正在修复模板...")

        # 原始模式：Access denied with code <*> Error reading POST data
        # 目标模式：Access denied with code <*>. Error reading POST data
        original_pattern = "Access denied with code <*> Error reading POST data"
        replacement_pattern = "Access denied with code <*>. Error reading POST data"

        # 检查是否包含待替换的模式
        if original_pattern in original_template:
            # 执行替换（只替换这一处）
            new_template = original_template.replace(
                original_pattern,
                replacement_pattern
            )

            df.loc[e1_index, 'EventTemplate'] = new_template

            print(f"✓ 替换成功")
            print(f"\n✅ 修复后的模板:")
            print(f"  {new_template}")

            # 4. 验证替换
            if "Access denied with code <*>. Error reading POST data" in new_template:
                print(f"\n✓ 验证通过：<*> 已正确替换为 <*>.")
            else:
                print(f"\n⚠️  验证失败：替换可能未成功")
                return False, 0

        else:
            print(f"⚠️  模板中未找到待替换的模式")
            print(f"   待找模式: {original_pattern}")
            return False, 0

        # 5. 备份原文件
        if backup:
            backup_file = input_file.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_file):
                import shutil
                shutil.copy(input_file, backup_file)
                print(f"\n💾 原文件已备份: {os.path.basename(backup_file)}")

        # 6. 保存修复后的模板文件
        print(f"\n💾 正在保存修复后的模板文件...")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 保存成功: {os.path.basename(output_file)}")

        return True, 1  # 返回成功标志和修改计数

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


# ==================== 第二步：同步修复结构化日志数据 ====================
def fix_structured_data(structured_file, templates_file, output_file, backup=True):
    """
    同步修复结构化日志数据

    1. 读取新的模板文件
    2. 找到使用 E1 模板且模式为 "Access denied with code <*> Error reading POST data" 的日志
    3. 更新这些日志的标签/内容以匹配新的模板格式
    """
    print(f"\n{'=' * 70}")
    print(f"📝 第二步：同步修复结构化日志数据")
    print(f"{'=' * 70}")

    if not os.path.exists(structured_file) or not os.path.exists(templates_file):
        print(f"❌ 文件不存在")
        return False, 0

    try:
        # 1. 读取修复后的模板文件
        print(f"📖 正在读取修复后的模板文件...")
        templates_df = pd.read_csv(templates_file, encoding='utf-8')

        # 获取 E1 的新模板
        e1_template = templates_df[templates_df['EventId'] == 'E1']['EventTemplate'].values[0]
        print(f"✓ E1 新模板: {e1_template}")

        # 2. 读取结构化日志文件
        print(f"\n📖 正在读取结构化日志文件...")
        logs_df = pd.read_csv(structured_file, encoding='utf-8', on_bad_lines='skip')
        print(f"✓ 读取成功，共 {len(logs_df)} 行日志")

        # 3. 检查列名
        print(f"\n📋 日志文件列名: {list(logs_df.columns)}")

        # 根据实际的列名来确定用哪一列作为模板 ID（通常是 'EventId' 或 'EventTemplate'）
        template_col = None
        if 'EventId' in logs_df.columns:
            template_col = 'EventId'
        elif 'EventTemplate' in logs_df.columns:
            template_col = 'EventTemplate'
        else:
            print(f"⚠️  找不到模板相关列，可用列: {list(logs_df.columns)}")
            return False, 0

        print(f"✓ 使用模板列: {template_col}")

        # 4. 找到所有 E1 类型的日志
        print(f"\n🔍 查找使用 E1 模板的日志...")
        e1_logs = logs_df[logs_df[template_col] == 'E1'] if template_col == 'EventId' else \
            logs_df[logs_df[template_col].str.contains('Access denied with code', na=False)]

        print(f"✓ 找到 {len(e1_logs)} 条 E1 类型的日志")

        if len(e1_logs) == 0:
            print(f"⚠️  没有 E1 类型的日志需要更新")
            return True, 0

        # 5. 显示修改前的样本
        print(f"\n📊 修改前的日志样本（前 3 条）:")
        for idx, (i, row) in enumerate(e1_logs.head(3).iterrows()):
            if idx < 3:
                print(f"  [{idx + 1}] {dict(row)}")

        # 6. 备份原文件
        if backup:
            backup_file = structured_file.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_file):
                import shutil
                shutil.copy(structured_file, backup_file)
                print(f"\n💾 原文件已备份: {os.path.basename(backup_file)}")

        # 7. 如果有 Content/Message 列，更新其中的模式
        update_count = 0
        if 'Content' in logs_df.columns or 'Message' in logs_df.columns:
            content_col = 'Content' if 'Content' in logs_df.columns else 'Message'
            print(f"\n🔧 正在更新日志内容（{content_col} 列）...")

            # 更新所有 E1 日志中的 "Access denied with code <*> Error reading POST data" 模式
            mask = logs_df[template_col] == 'E1'

            def update_content(text):
                if pd.isna(text):
                    return text
                # 替换模式中的 <*> 为 <*>.
                text_str = str(text)
                if "Access denied with code <*> Error reading POST data" in text_str:
                    return text_str.replace(
                        "Access denied with code <*> Error reading POST data",
                        "Access denied with code <*>. Error reading POST data"
                    )
                return text_str

            logs_df.loc[mask, content_col] = logs_df.loc[mask, content_col].apply(update_content)

            # 计算更新数量
            update_count = (logs_df.loc[mask, content_col].astype(str).str.contains(
                'Access denied with code <\*>\. Error reading POST data', na=False, regex=True)).sum()
            print(f"✓ 更新了 {update_count} 条日志")

        # 8. 显示修改后的样本
        e1_logs_updated = logs_df[logs_df[template_col] == 'E1'].head(3)
        print(f"\n📊 修改后的日志样本（前 3 条）:")
        for idx, (i, row) in enumerate(e1_logs_updated.iterrows()):
            if idx < 3:
                print(f"  [{idx + 1}] {dict(row)}")

        # 9. 保存修复后的数据
        print(f"\n💾 正在保存修复后的日志数据...")
        logs_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 保存成功: {os.path.basename(output_file)}")

        return True, update_count

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


# ==================== 主函数 ====================
def main():
    """
    主函数：执行完整的修复流程
    """
    print("\n" + "=" * 70)
    print("🚀 Apache 日志通配符替换工具")
    print("=" * 70)
    print(f"当前用户: {CURRENT_USER}")
    print(f"当前时间: {TASK_START_TIME}")
    print(f"数据集目录: {DATASET_DIR}")
    print(f"\n任务说明:")
    print(f"  1. 修复模板文件中 E1 的通配符格式")
    print(f"     原: Access denied with code <*> Error reading POST data")
    print(f"     新: Access denied with code <*>. Error reading POST data")
    print(f"        ↑ 把 <*> 改成 <*>. ↑")
    print(f"  2. 同步修复结构化日志数据中对应的内容")

    # 检查目录和文件
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ 数据集目录不存在: {DATASET_DIR}")
        return

    if not os.path.exists(TEMPLATES_FILE):
        print(f"❌ 模板文件不存在: {TEMPLATES_FILE}")
        return

    if not os.path.exists(STRUCTURED_FILE):
        print(f"❌ 结构化日志文件不存在: {STRUCTURED_FILE}")
        return

    # 第一步：修复模板文件
    print("\n\n[1/2] 修复模板文件 >>>")
    template_success, template_count = fix_template_file(TEMPLATES_FILE, OUTPUT_TEMPLATES)

    # 第二步：同步修复结构化日志数据
    print("\n\n[2/2] 修复结构化日志数据 >>>")
    if template_success:
        # 使用修复后的模板文件
        data_success, data_count = fix_structured_data(STRUCTURED_FILE, OUTPUT_TEMPLATES, OUTPUT_STRUCTURED)
    else:
        print(f"❌ 模板文件修复失败，跳过数据修复步骤")
        data_success, data_count = False, 0

    # 总结
    print("\n\n" + "=" * 70)
    print("📋 处理总结")
    print("=" * 70)

    if template_success:
        print(f"✅ 模板文件修复成功")
        print(f"   输出: {OUTPUT_TEMPLATES}")
        print(f"   修改项数: {template_count}")
    else:
        print(f"❌ 模板文件修复失败")

    print()

    if data_success:
        print(f"✅ 结构化日志数据修复成功")
        print(f"   输出: {OUTPUT_STRUCTURED}")
        print(f"   修改项数: {data_count}")
    else:
        print(f"❌ 结构化日志数据修复失败")

    if template_success and data_success:
        print("\n🎉 所有修复完毕！")
        print(f"\n📌 生成的文件:")
        print(f"   1. {os.path.basename(OUTPUT_TEMPLATES)}")
        print(f"   2. {os.path.basename(OUTPUT_STRUCTURED)}")
        print(f"\n💾 原文件已备份为 _backup.csv")
    else:
        print("\n⚠️  部分修复失败，请检查错误信息")

    print("=" * 70 + "\n")


# ==================== 运行脚本 ====================
if __name__ == "__main__":
    main()