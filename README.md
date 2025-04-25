# CCRS_Library

**人工智能轮毂铸造字符识别 API 库**

## 概述

**CCRS_Library** 是一个专注于提供图像处理和字符识别 API 的开源库，专为人工智能轮毂铸造字符识别系统设计。该库通过深度学习模型（YOLOv5 和 YOLOv11）支持轮毂铸造字符的自动识别，适用于制造业的智能质量检测场景。库的模块化设计使其易于集成到自动化生产线上，提供高效的字符检测和识别功能。

本库基于 **GNU 宽松通用公共许可证 v3.0 (LGPLv3)** 发布，允许在开源和专有项目中使用，但对库本身的修改需遵循 LGPLv3 许可证。

## 功能特性

**CCRS_Library** 提供以下核心 API 接口：

- **图像裁剪 (**`clear_pic_5/clear_pic_11`**)**：

  - 裁剪轮毂图像中的模具编号区域，返回裁剪后的图像变量。
  - 支持 YOLOv5 和 YOLO 11模型（如 `NumVision.pt`）。

- **字符识别 (**`get_num`**)**：

  - 对裁剪后的图像进行模具编号识别，返回识别的文本。
  - 提供三种识别 API：
    - `get_num_cls`：基于 YOLO11-Cls 模型，高准确性，仅限训练数据集。
    - `get_num_obb`：基于 YOLO11-OBB 模型，支持倾斜图像，灵活但速度较慢。
    - `get_num_obj`：基于 YOLO11 模型，平衡速度和准确性。

- **数据库操作 (**`sql`**)**：

  - 提供便捷的数据库访问接口，支持执行自定义 SQL 命令（需外部实现安全检查）。

- **系统信息 (**`sys_info`**)**：

  - 获取服务器系统信息并返回给客户端，用于监控和调试。

## 安装

### 前提条件

- **Python**：3.8 或更高版本
- **依赖库**：
  - `ultralytics`（用于 YOLOv11 模型）
  - `torch`（用于 YOLOv5 和 YOLOv11 模型）
  - `opencv-python`（用于图像处理）
  - 其他依赖见 `requirements.txt`
- **硬件要求**：
  - CPU：四核 2.40GHz 或更高
  - 内存：32GB 或更多
  - 存储：至少 10GB 可用空间

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone https://github.com/xin1201946/CRS_Library.git
   cd ccrs-library
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 示例：图像裁剪与字符识别

```python
from ccrs_library import clear_pic_11, get_num_cls

# 裁剪模具编号区域，返回图片变量以及缓存路径
img, paths = clear_pic_11(get_pic(path))

# 识别模具编号
mold_number = get_num_cls(path)
print(f"识别的模具编号: {mold_number}")
```

### API 接口

**CCRS_Library** 提供的 API 接口在 `getNum.py` 中定义，具体功能如下：

- **图像裁剪**：裁剪图像并返回裁剪后的图像变量。

  - `clear_pic_11`：Yolo11
  - `clear_pic_5`: Yolo v5

- **字符识别**：

  - `get_num_cls`：高准确性识别（YOLO11-Cls，`num_class_v2.pt`）。
  - `get_num_obb`：支持倾斜图像的灵活识别（YOLO11-OBB，`num_obb_v2.pt`）。
  - `get_num_obj`：平衡速度和准确性的识别（YOLO11，`num_obj_v1.pt`）。

- **数据库操作**：

  - `sql`：执行自定义 SQL 命令，需外部实现安全验证。

- **系统信息**：

  - `sys_info`：返回服务器运行状态和信息。

### 注意事项

- **首次识别延迟**：首次调用 API 时可能因动态加载模型而稍慢，后续调用会因缓存而加速。

## 许可证

本库采用 **GNU 宽松通用公共许可证 v3.0 (LGPLv3)** 发布。您可以自由使用、修改和分发本库，但需遵守以下条件：

- 对库本身的任何修改必须以 LGPLv3 许可证发布。
- 使用本库（静态或动态链接）的应用程序可采用任意许可证（包括专有许可证），但需尊重本库的许可证条款。

详情请参阅 LICENSE 文件。

## 贡献

欢迎为本项目贡献代码！贡献步骤如下：

1. Fork 本仓库。
2. 创建新分支（`git checkout -b feature/your-feature`）。
3. 提交更改（`git commit -m '添加您的功能'`）。
4. 推送分支（`git push origin feature/your-feature`）。
5. 提交 Pull Request。

请确保代码符合项目编码规范并包含适当的测试。

## 支持

如有问题、疑问或功能请求，请在 GitHub 仓库 提交 issue。如需更多支持，请参阅下面的 常见问题解答 部分。

## 常见问题解答

1. **为什么第一次调用 API 速度慢？**

   - 库在首次调用时动态加载模型以优化启动时间。后续调用因模型缓存而变快。

2. **库支持哪些模型？**

   - 图像裁剪支持 YOLOv5和YOLO11，字符识别支持 YOLO11-Cls（`num_class_v2.pt`）、YOLO11-OBB（`num_obb_v2.pt`）和 YOLO11（`num_obj_v1.pt`）。

## 致谢

- 感谢开源社区的贡献以及 Ultralytics 提供的 YOLO 模型。

---

*本库是(CRS/CCRS)[https://github.com/xin1201946/CRS_Python]的一部分，旨在通过高效的 API 支持制造业智能质量检测的创新。*
