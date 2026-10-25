import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
import shutil
import random
from pathlib import Path
import pickle


def create_augmentation_pipeline():
    """数据增强"""
    seq = iaa.Sequential([
        # 仿射变换
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 随机缩放
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 随机平移
            rotate=(-30, 30),  # 随机旋转
            shear=(-10, 10),  # 随机剪切
            order=[0, 1],  # 使用最近邻或双线性插值
            cval=(0, 255),  # 填充值范围
            mode='reflect'  # 边缘填充模式
        ),
        # 随机水平翻转
        iaa.Fliplr(0.5),
        # 随机垂直翻转
        iaa.Flipud(0.3),
        # 随机裁剪
        iaa.Sometimes(
            0.3,
            iaa.Crop(percent=(0, 0.1))
        ),
        # 随机对比度调整
        iaa.LinearContrast((0.8, 1.2)),
        # 随机亮度调整
        iaa.MultiplyBrightness((0.8, 1.2)),
        # 随机添加高斯噪声
        iaa.Sometimes(
            0.3,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255))
        ),
        # 随机模糊
        iaa.Sometimes(
            0.3,
            iaa.GaussianBlur(sigma=(0, 1.0))
        ),
        # 随机调整饱和度
        iaa.Sometimes(
            0.3,
            iaa.MultiplySaturation((0.8, 1.2))
        )
    ], random_order=True)

    return seq


def load_pickle_data(pickle_path):
    """加载pkl文件中的数据（如果存在）"""
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    return None


def save_pickle_data(data, pickle_path):
    """保存数据到pkl文件"""
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def augment_images(source_root, target_root, augmentations_per_image=5):
    """
    对源文件夹中的所有图片进行数据增强

    参数:
    source_root: 源文件夹路径
    target_root: 目标文件夹路径
    augmentations_per_image: 每张原始图片生成的增强图片数量
    """

    # 创建数据增强管道
    augmenter = create_augmentation_pipeline()

    # 获取源pkl文件路径
    source_pickle = os.path.join(source_root, "emotion_dataset_enhanced.pkl")
    target_pickle = os.path.join(target_root, "emotion_dataset_strong.pkl")

    # 加载原始pkl数据
    original_data = load_pickle_data(source_pickle)
    augmented_data = []

    # 获取所有情绪类别文件夹
    emotion_folders = [f for f in os.listdir(source_root)
                       if os.path.isdir(os.path.join(source_root, f))]

    print(f"找到 {len(emotion_folders)} 个情绪类别文件夹")
    print("情绪类别:", emotion_folders)

    total_original = 0
    total_augmented = 0

    for emotion in emotion_folders:
        source_emotion_dir = os.path.join(source_root, emotion)
        target_emotion_dir = os.path.join(target_root, emotion)

        # 创建目标文件夹
        os.makedirs(target_emotion_dir, exist_ok=True)

        # 获取源文件夹中的所有图片
        image_files = [f for f in os.listdir(source_emotion_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        print(f"\n处理情绪类别: {emotion}")
        print(f"原始图片数量: {len(image_files)}")

        for img_file in image_files:
            source_img_path = os.path.join(source_emotion_dir, img_file)

            # 读取图片
            img = cv2.imread(source_img_path)
            if img is None:
                print(f"无法读取图片: {source_img_path}")
                continue

            # 保存原始图片到目标文件夹
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            original_target_path = os.path.join(target_emotion_dir, f"{base_name}_original{ext}")
            cv2.imwrite(original_target_path, img)

            # 如果存在pkl数据，复制对应条目
            if original_data:
                # 这里需要根据你的pkl数据结构来调整
                # 假设pkl数据是列表，每个元素包含路径和标签
                pass

            total_original += 1

            # 生成增强图片
            for i in range(augmentations_per_image):
                # 应用数据增强
                img_augmented = augmenter.augment_image(img)

                # 保存增强后的图片
                aug_img_name = f"{base_name}_aug_{i + 1}{ext}"
                aug_img_path = os.path.join(target_emotion_dir, aug_img_name)
                cv2.imwrite(aug_img_path, img_augmented)

                # 如果存在pkl数据，添加增强图片的条目
                if original_data:
                    # 这里需要根据你的pkl数据结构来调整
                    pass

                total_augmented += 1

            if total_original % 10 == 0:
                print(f"已处理 {total_original} 张原始图片，生成 {total_augmented} 张增强图片")

    # 如果有pkl数据，保存增强后的数据
    if original_data and augmented_data:
        save_pickle_data(augmented_data, target_pickle)

    print(f"\n数据增强完成！")
    print(f"原始图片总数: {total_original}")
    print(f"生成的增强图片总数: {total_augmented}")
    print(f"增强后总图片数: {total_original + total_augmented}")
    print(f"增强倍数: {augmentations_per_image} 倍")
    print(f"目标文件夹: {target_root}")


def visualize_augmentations(image_path, num_augmentations=5):
    """
    可视化数据增强效果

    参数:
    image_path: 要可视化的图片路径
    num_augmentations: 生成的增强图片数量
    """
    import matplotlib.pyplot as plt

    # 创建数据增强管道
    augmenter = create_augmentation_pipeline()

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 转换BGR到RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # 显示原始图片
    axes[0].imshow(img_rgb)
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    # 生成并显示增强图片
    for i in range(1, 6):
        img_augmented = augmenter.augment_image(img)
        img_augmented_rgb = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_augmented_rgb)
        axes[i].set_title(f"增强 {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 根据您的截图，修正路径
    # 您的截图显示路径是：C:\Users\M\Desktop\face_emotion_datasets
    # emotion_dataset_enhanced文件夹直接在此路径下
    base_path = r"C:\Users\M\Desktop\face_emotion_datasets\dataset_clean"
    source_folder = os.path.join(base_path, "emotion_dataset_enhanced")
    target_folder = os.path.join(base_path, "emotion_dataset_strong")

    print("检查源文件夹路径...")
    print(f"源文件夹: {source_folder}")
    print(f"是否存在: {os.path.exists(source_folder)}")

    if os.path.exists(source_folder):
        print("源文件夹内容:")
        for item in os.listdir(source_folder):
            item_path = os.path.join(source_folder, item)
            if os.path.isdir(item_path):
                print(f"  文件夹: {item} (包含 {len(os.listdir(item_path))} 个文件)")
            else:
                print(f"  文件: {item}")

    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹不存在: {source_folder}")
        print(f"请检查路径是否正确。根据您的截图，正确的路径应该是:")
        print(f"C:\\Users\\M\\Desktop\\face_emotion_datasets\\emotion_dataset_enhanced")
        return

    # 如果目标文件夹已存在，询问是否删除
    if os.path.exists(target_folder):
        response = input(f"目标文件夹 {target_folder} 已存在。要删除并重新创建吗？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(target_folder)
            print(f"已删除旧文件夹: {target_folder}")
        else:
            print("操作已取消。")
            return

    print("\n开始数据增强处理...")
    print(f"源文件夹: {source_folder}")
    print(f"目标文件夹: {target_folder}")

    # 设置每张图片的增强数量
    augmentations_per_image = 5  # 每张原始图片生成5张增强图片

    # 执行数据增强
    augment_images(source_folder, target_folder, augmentations_per_image)

    # 可选：可视化增强效果
    visualize_choice = input("\n是否要可视化增强效果？(y/n): ")
    if visualize_choice.lower() == 'y':
        # 从任意文件夹中取一张图片来可视化
        emotion_folders = [f for f in os.listdir(source_folder)
                           if os.path.isdir(os.path.join(source_folder, f))]
        if emotion_folders:
            sample_folder = os.path.join(source_folder, emotion_folders[0])
            image_files = [f for f in os.listdir(sample_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                sample_image = os.path.join(sample_folder, image_files[0])
                visualize_augmentations(sample_image)

    print("\n数据增强处理完成！")


if __name__ == "__main__":
    # 检查并安装必要的库
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行以下命令安装：")
        print("pip install opencv-python numpy")
        print("然后重新运行此脚本。")
        exit(1)

    # 检查imgaug库
    try:
        from imgaug import augmenters as iaa
    except ImportError:
        print("缺少imgaug库，正在尝试安装...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "imgaug"])
        from imgaug import augmenters as iaa

    # 检查matplotlib库
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("缺少matplotlib库，正在尝试安装...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt

    main()