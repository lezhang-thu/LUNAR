import os
import csv
from collections import defaultdict
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================
DATASET_PATH = r"E:\LUNAR-THU\datasets\Hadoop"
TEMPLATES_FILE = os.path.join(DATASET_PATH, "Hadoop_full.log_structured.csv")
BACKUP_DIR = os.path.join(DATASET_PATH, "backups")
OUTPUT_DIR = os.path.join(DATASET_PATH, "fixed_templates")
REPORT_DIR = os.path.join(DATASET_PATH, "reports")

# ============================================================================
# 修复规则 - 精确定义（包含正确的规则6）
# ============================================================================
TEMPLATE_FIXES = [
    {
        "id": 1,
        "name": "内存和核心格式修复",
        "find_exact": "<memory:<*>, vCores:<*>",
        "replace_with": "<memory:<*>, vCores:<*>>",
        "explanation": "添加缺失的右尖括号"
    },
    {
        "id": 2,
        "name": "本地主机地址格式修复",
        "find_exact": "local host is: <*> destination host is: <*>",
        "replace_with": "local host is: <*> destination host is: <*>;",
        "explanation": "添加缺失的分号"
    },
    {
        "id": 3,
        "name": "多容器作业标识修复",
        "find_exact": "multi-container job <*>",
        "replace_with": "multi-container job <*>.",
        "explanation": "添加缺失的句点"
    },
    {
        "id": 4,
        "name": "Socket Reader 端口格式修复",
        "find_exact": "Socket Reader <*> for port <*> readAndProcess",
        "replace_with": "Socket Reader <*> for port <*>:",
        "explanation": "将 'readAndProcess' 替换为冒号"
    },
    {
        "id": 5,
        "name": "资源释放容器信息修复",
        "find_exact": "Releasing unassigned and invalid container Container: [ContainerId: <*>, NodeId: <*>, NodeHttpAddress: <*>, Resource: <*>, Priority: <*>, Token: <*>]. RM may have assignment issues",
        "replace_with": "Releasing unassigned and invalid container Container: [ContainerId: <*>, NodeId: <*>, NodeHttpAddress: <*>, Resource: <memory:<*>, vCores:<*>>, Priority: <*>, Token: Token { kind: <*>, service: <*> }, ]. RM may have assignment issues",
        "explanation": "修复 Resource 和 Token 字段结构"
    },
    {
        "id": 6,
        "name": "Fetcher shuffle 输出格式修复",
        "find_exact": "fetcher#<*> about to shuffle output of map <*>: <*> len: <*> to DISK",
        "replace_with": "fetcher#<*> about to shuffle output of map <*> decomp: <*> len: <*> to DISK",
        "explanation": "修改格式：删除第二个冒号，添加 decomp 字段"
    },
]

# ============================================================================
# 颜色工具
# ============================================================================
class Colors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    LIGHT_GRAY = '\033[37m'


def print_header(text, char="="):
    width = 160
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{char * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{char * width}{Colors.ENDC}\n")


def print_section(text, char="-"):
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{char * 160}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{char * 160}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")


# ============================================================================
# 核心功能
# ============================================================================

def backup_file(file_path):
    """创建文件备份"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"Hadoop_full.log_structured_backup_{timestamp}.csv")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as src:
                content = src.read()
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
            print_success(f"备份文件已创建: {backup_path}")
            return backup_path
        except Exception as e:
            print_error(f"备份失败: {e}")
            return None
    return None


def read_structured_file(file_path):
    """读取 CSV 文件"""
    data = []
    columns = []

    if not os.path.exists(file_path):
        print_error(f"文件不存在: {file_path}")
        return data, columns

    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            columns = next(reader)
            for row in reader:
                data.append(row)
        print_success(f"已读取 {len(data)} 行数据，{len(columns)} 列")
        return data, columns
    except Exception as e:
        print_error(f"读取文件错误: {e}")
        return data, columns


def apply_fixes_to_template(template_text: str):
    """应用所有修复规则到一个模板"""
    fixes_applied = []
    modified_text = template_text

    for fix in TEMPLATE_FIXES:
        find_exact = fix["find_exact"]
        replace_with = fix["replace_with"]

        # 如果模板中包含要查找的模式，就进行替换
        if find_exact in modified_text:
            new_text = modified_text.replace(find_exact, replace_with)

            if new_text != modified_text:
                fixes_applied.append({
                    "id": fix["id"],
                    "name": fix["name"],
                    "find": find_exact,
                    "replace": replace_with,
                    "before": modified_text,
                    "after": new_text
                })
                modified_text = new_text

    return modified_text, fixes_applied


def save_structured_file(output_path: str, columns: list, data: list):
    """保存为新 CSV 文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in data:
                writer.writerow(row)
        print_success(f"新文件已保存: {output_path}")
        return True
    except Exception as e:
        print_error(f"保存文件错误: {e}")
        return False


def generate_report(original_data, modified_data, columns, template_col_idx, all_changes):
    """生成详细修复报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    txt_report_path = os.path.join(REPORT_DIR, f"structured_fixes_report_{timestamp}.txt")

    # 统计规则使用
    rule_usage = defaultdict(int)
    for row_idx, changes in all_changes.items():
        for fix in changes:
            rule_usage[fix['id']] += 1

    try:
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 180 + "\n")
            f.write("Hadoop Structured 日志文件修复详细报告\n")
            f.write("=" * 180 + "\n\n")

            f.write(f"执行时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"操作用户: XiancongMeng\n")
            f.write(f"数据集路径: {DATASET_PATH}\n")
            f.write(f"原始文件: {TEMPLATES_FILE}\n\n")

            total_rows = len(original_data)
            modified_count = len(all_changes)

            f.write("-" * 180 + "\n")
            f.write("修复统计\n")
            f.write("-" * 180 + "\n")
            f.write(f"总行数:       {total_rows:,}\n")
            f.write(f"已修改:       {modified_count:,}\n")
            f.write(f"未修改:       {total_rows - modified_count:,}\n")
            f.write(f"修改率:       {(modified_count / total_rows * 100):.4f}%\n\n")

            f.write("-" * 180 + "\n")
            f.write("规则使用统计\n")
            f.write("-" * 180 + "\n")
            for fix in TEMPLATE_FIXES:
                count = rule_usage.get(fix['id'], 0)
                f.write(f"规则 {fix['id']}: {fix['name']} - {count:,} 次\n")
            f.write("\n")

            f.write("-" * 180 + "\n")
            f.write("详细修改记录（前 200 条）\n")
            f.write("-" * 180 + "\n\n")

            for idx, (row_idx, changes) in enumerate(list(all_changes.items())[:200], 1):
                original_row = original_data[int(row_idx)]
                modified_row = modified_data[int(row_idx)]

                original_template = original_row[template_col_idx]
                modified_template = modified_row[template_col_idx]

                f.write(f"\n修改 {idx}. 行 {row_idx}:\n")
                f.write(f"  修改前: {original_template}\n")
                f.write(f"  修改后: {modified_template}\n")
                f.write(f"  应用规则 ({len(changes)} 个):\n")
                for fix in changes:
                    f.write(f"    - 规则 {fix['id']}: {fix['name']}\n")

            if len(all_changes) > 200:
                f.write(f"\n... 还有 {len(all_changes) - 200:,} 条修改记录\n")

            f.write("\n" + "=" * 180 + "\n")
            f.write("报告完成\n")
            f.write("=" * 180 + "\n")

        print_success(f"详细报告已生成: {txt_report_path}")
        return txt_report_path
    except Exception as e:
        print_error(f"生成报告失败: {e}")
        return None


# ============================================================================
# 主函数
# ============================================================================

def main():
    print_header("Hadoop Structured 日志文件修复工具 - 包含规则6", "=")

    print_info(f"执行时间: 2025-11-10 04:00:15 UTC")
    print_info(f"用户: XiancongMeng")
    print_info(f"数据集: {DATASET_PATH}\n")

    if not os.path.exists(TEMPLATES_FILE):
        print_error(f"文件不存在: {TEMPLATES_FILE}")
        return

    # 创建备份
    print_section("第一步：创建备份", "-")
    backup_file(TEMPLATES_FILE)

    # 读取文件
    print_section("第二步：读取文件", "-")
    original_data, columns = read_structured_file(TEMPLATES_FILE)
    if not original_data:
        return

    template_col_idx = len(columns) - 1
    print_success(f"模板列: [{template_col_idx}] {columns[template_col_idx]}\n")

    # 显示修复规则
    print_section("修复规则列表", "-")
    for fix in TEMPLATE_FIXES:
        print(f"{Colors.BOLD}规则 {fix['id']}: {fix['name']}{Colors.ENDC}")
        print(f"  查找:  '{fix['find_exact']}'")
        print(f"  替换为: '{fix['replace_with']}'")
        print(f"  说明:  {fix['explanation']}\n")

    # 应用修复
    print_section("第三步：应用修复规则", "-")
    print_info("正在扫描和修复所有模板...\n")

    modified_data = [row[:] for row in original_data]
    all_changes = defaultdict(list)

    for row_idx, row in enumerate(original_data):
        if template_col_idx < len(row):
            original_template = row[template_col_idx]
            modified_template, fixes_applied = apply_fixes_to_template(original_template)

            if fixes_applied:
                modified_data[row_idx][template_col_idx] = modified_template
                all_changes[str(row_idx)] = fixes_applied

                # 打印修改详情
                print(f"{Colors.BOLD}行 {row_idx}:{Colors.ENDC}")
                for fix in fixes_applied:
                    print(f"  {Colors.OKGREEN}✓ 规则 {fix['id']}: {fix['name']}{Colors.ENDC}")
                    print(f"    {Colors.WARNING}修改前:{Colors.ENDC} {fix['before']}")
                    print(f"    {Colors.OKGREEN}修改后:{Colors.ENDC} {fix['after']}\n")

    # 统计
    print_section("修复完成统计", "=")
    total_rows = len(original_data)
    modified_count = len(all_changes)

    print(f"{Colors.BOLD}基本统计:{Colors.ENDC}")
    print(f"  总行数:     {total_rows:,}")
    print(f"  {Colors.OKGREEN}已修改:     {modified_count:,}{Colors.ENDC}")
    print(f"  {Colors.WARNING}未修改:     {total_rows - modified_count:,}{Colors.ENDC}")
    print(f"  修改率:     {(modified_count / total_rows * 100):.4f}%\n")

    # 规则统计
    rule_usage = defaultdict(int)
    for row_idx, changes in all_changes.items():
        for fix in changes:
            rule_usage[fix['id']] += 1

    print(f"{Colors.BOLD}规则使用统计:{Colors.ENDC}")
    for fix in TEMPLATE_FIXES:
        count = rule_usage.get(fix['id'], 0)
        status = Colors.OKGREEN if count > 0 else Colors.WARNING
        print(f"  规则 {fix['id']}: {status}{count:,}{Colors.ENDC} 次 - {fix['name']}")

    # 用户确认
    print_section("第四步：用户确认", "-")
    if modified_count > 0:
        print_warning(f"即将修改 {modified_count:,} 行数据")
    else:
        print_warning("未检测到需要修改的数据")

    response = input(f"\n{Colors.BOLD}是否继续保存修改? (y/n): {Colors.ENDC}").strip().lower()

    if response == 'y':
        # 保存新文件
        print_section("第五步：保存文件", "-")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"Hadoop_full.log_structured_fixed_{timestamp}.csv")

        if save_structured_file(output_file, columns, modified_data):
            # 生成报告
            print_section("第六步：生成报告", "-")
            txt_report = generate_report(original_data, modified_data, columns, template_col_idx, all_changes)

            # 最终总结
            print_section("✅ 修复成功完成！", "=")
            print(f"{Colors.OKGREEN}✓ 原始文件: {TEMPLATES_FILE} (未修改){Colors.ENDC}")
            print(f"{Colors.OKGREEN}✓ 新文件: {output_file}{Colors.ENDC}")
            if txt_report:
                print(f"{Colors.OKGREEN}✓ 报告: {txt_report}{Colors.ENDC}")

            print(f"\n{Colors.BOLD}最终统计:{Colors.ENDC}")
            print(f"  总行数: {total_rows:,}")
            print(f"  修改行数: {modified_count:,}")
            print(f"  修改率: {(modified_count / total_rows * 100):.4f}%")
    else:
        print_error("操作已取消")


if __name__ == "__main__":
    main()