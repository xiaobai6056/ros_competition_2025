import os
import shutil
import random

# -------------------------- 无需修改的参数（适配当前目录） --------------------------
# 原始图像和标签路径（当前目录下的 Images 和 Labels）
img_dir = os.path.join(os.getcwd(), "Images")
label_dir = os.path.join(os.getcwd(), "Labels")
# 划分后保存路径（当前目录下的 train 和 val）
train_img_save = os.path.join(os.getcwd(), "train", "images")
train_label_save = os.path.join(os.getcwd(), "train", "labels")
val_img_save = os.path.join(os.getcwd(), "val", "images")
val_label_save = os.path.join(os.getcwd(), "val", "labels")
# 划分比例：80% 训练集，20% 验证集
train_ratio = 0.9
# -----------------------------------------------------------------------------------

# 1. 获取所有图像文件（只保留 .jpg/.png 格式）
img_files = []
for f in os.listdir(img_dir):
    if f.endswith((".jpg", ".png", ".jpeg")):  # 覆盖常见图像格式
        img_files.append(f)

# 2. 检查是否有图像文件
if len(img_files) == 0:
    print("错误：Images 文件夹下没有找到 .jpg/.png 图像文件！")
    exit()

# 3. 打乱图像顺序（固定随机种子，确保每次划分结果一致）
random.seed(42)
random.shuffle(img_files)

# 4. 计算划分数量（比如 100 张图：80 张训练，20 张验证）
total_num = len(img_files)
train_num = int(total_num * train_ratio)
val_num = total_num - train_num

print(f"总共 {total_num} 张图像，划分结果：")
print(f"训练集：{train_num} 张，验证集：{val_num} 张")

# 5. 复制训练集（图像 + 对应标签）
for img_file in img_files[:train_num]:
    # 复制图像
    img_src = os.path.join(img_dir, img_file)
    img_dst = os.path.join(train_img_save, img_file)
    shutil.copy(img_src, img_dst)
    # 复制对应标签（图像后缀改成 .txt）
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_src = os.path.join(label_dir, label_file)
    # 检查标签是否存在，存在才复制（避免报错）
    if os.path.exists(label_src):
        label_dst = os.path.join(train_label_save, label_file)
        shutil.copy(label_src, label_dst)
    else:
        print(f"警告：训练集标签 {label_file} 不存在，已跳过")

# 6. 复制验证集（和训练集逻辑一样）
for img_file in img_files[train_num:]:
    img_src = os.path.join(img_dir, img_file)
    img_dst = os.path.join(val_img_save, img_file)
    shutil.copy(img_src, img_dst)
    
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_src = os.path.join(label_dir, label_file)
    if os.path.exists(label_src):
        label_dst = os.path.join(val_label_save, label_file)
        shutil.copy(label_src, label_dst)
    else:
        print(f"警告：验证集标签 {label_file} 不存在，已跳过")

print("数据集划分完成！")
# 打印划分后的数据量，确认是否正确
print(f"训练集图像数量：{len(os.listdir(train_img_save))}")
print(f"训练集标签数量：{len(os.listdir(train_label_save))}")
print(f"验证集图像数量：{len(os.listdir(val_img_save))}")
print(f"验证集标签数量：{len(os.listdir(val_label_save))}")

