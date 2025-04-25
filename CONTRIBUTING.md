# CCRS_Library 社区贡献指南

欢迎加入 **CCRS_Library** 的开源社区！**CCRS_Library** 是一个专注于人工智能轮毂铸造字符识别的 API 库，旨在通过高效的图像处理和字符识别功能推动制造业智能质量检测的创新。我们非常感谢您的兴趣，并期待您通过贡献代码、文档、测试或其他方式帮助改进项目。

本指南将帮助您了解如何参与贡献，确保您的贡献符合项目标准，并与社区协作顺畅。本项目遵循 **GNU 宽松通用公共许可证 v3.0 (LGPLv3)**，请确保您的贡献符合许可证要求。

## 如何开始贡献

### 1. 了解项目

在开始贡献之前，请阅读以下文档以熟悉项目：

- **README.md**：了解 **CCRS_Library** 的功能、安装方法和 API 使用方式。
- **项目代码结构**：查看 `getNum.py` 和其他核心文件，了解 API 实现（如 `clear_pic`、`get_num_cls` 等）。
- **问题跟踪**：浏览 GitHub Issues 以了解当前的需求、错误报告和功能请求。

### 2. 选择贡献方式

您可以通过以下方式为 **CCRS_Library** 做出贡献：

- **修复 Bug**：解决 GitHub Issues 中标记为 `bug` 的问题。
- **实现新功能**：提出并实现新 API 或优化现有功能（如改进字符识别算法）。
- **改进文档**：完善 README、API 文档或添加使用示例。
- **编写测试**：为 API 接口（如 `get_num_obb`、`sql`）添加单元测试。
- **优化性能**：提高图像处理或字符识别的效率。
- **报告问题**：提交详细的 Bug 报告或功能建议。

### 3. 设置开发环境

按照 README.md 的安装步骤设置开发环境：

1. 克隆仓库：

   ```bash
   git clone https://github.com/xin1201946/CRS_Library.git
   cd CRS_Library
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 确保模型文件（如 `num_class_v2.pt`）可用，并配置测试环境。

## 贡献流程

### 1. 查找或创建 Issue

- 在 GitHub Issues 中查找您想解决的问题，或创建一个新 Issue 描述您的贡献计划。
- 如果是新功能，请详细说明功能用途、实现方式和潜在影响。
- 等待 maintainers 确认或分配 Issue 给您。

### 2. Fork 和创建分支

1. Fork 仓库到您的 GitHub 账户。

2. 克隆您的 Fork：

   ```bash
   git clone https://github.com/xin1201946/CRS_Library.git
   ```

3. 创建一个描述性分支：

   ```bash
   git checkout -b feature/your-feature-name
   ```

   或

   ```bash
   git checkout -b bugfix/issue-number-description
   ```

   示例：`feature/add-get-num-new-api` 或 `bugfix/123-fix-sql-injection`。

### 3. 开发和提交代码

- **编码规范**：

  - 遵循 Python PEP 8 编码风格。
  - 为每个 API 函数添加清晰的注释和文档字符串。
  - 确保代码模块化，避免修改非必要部分。

- **测试**：

  - 为新功能或修复添加单元测试（使用 `unittest` 或 `pytest`）。
  - 验证 API 在不同模型（如 YOLOv5 和 YOLOv11）上的兼容性。
  - 测试环境：Python 3.8+，确保依赖（如 `ultralytics`、`torch`）正常工作。

- **提交更改**：

  - 使用清晰的提交信息，遵循格式：

    ```
    <类型>(<范围>): <简短描述>
    <空行>
    <详细说明（可选）>
    ```

    示例：

    ```
    feat(get_num): add support for new YOLO11 model
    Added get_num_new API to support custom YOLO11 models for improved flexibility.
    ```

    类型包括：`feat`（新功能）、`fix`（修复）、`docs`（文档）、`test`（测试）、`refactor`（重构）等。

  - 提交代码：

    ```bash
    git add .
    git commit -m "feat(get_num): add support for new YOLO11 model"
    ```

### 4. 推送和提交 Pull Request (PR)

1. 推送分支到您的 Fork：

   ```bash
   git push origin feature/your-feature-name
   ```

2. 在 GitHub 上创建 Pull Request：

   - 选择您的分支与主仓库的 `main` 分支合并。
   - 填写 PR 模板（如果有），包括：
     - 关联的 Issue 编号（例如 `Fixes #123`）。
     - 更改描述和测试验证方式。
     - 对 LGPLv3 许可证的遵守声明。
   - 示例 PR 标题：`Add get_num_new API for custom YOLO11 models (#123)`

3. 等待代码审查，回应 maintainers 的反馈并进行必要修改。

### 5. 代码审查和合并

- Maintainers 将审查您的 PR，确保代码质量、功能正确性和许可证合规性。
- 根据反馈修改代码并更新 PR。
- 一旦通过审查，您的代码将被合并到主分支。

## 社区行为准则

我们致力于打造一个包容、友好的社区环境。请遵守以下准则：

- **尊重他人**：以礼貌和建设性的方式沟通，避免任何形式的歧视或骚扰。
- **开放协作**：欢迎不同背景和经验的贡献者，鼓励分享想法。
- **报告问题**：如遇到不当行为，请通过 GitHub Issues 或私下联系 maintainers 报告。

## LGPLv3 许可证要求

**CCRS_Library** 遵循 **GNU 宽松通用公共许可证 v3.0 (LGPLv3)**。贡献者需注意：

- 您提交的代码将以 LGPLv3 许可证发布。
- 如果修改了库的源代码，修改后的版本必须以 LGPLv3 许可证开源。
- 使用 **CCRS_Library** 的应用程序可以采用任何许可证（包括专有许可证），但需提供库的源代码及其修改版本（如果有）。

## 常见问题解答

1. **我需要编写测试吗？**
   - 是的，所有新功能和 Bug 修复都应包含单元测试，以确保 API 的稳定性（如 `get_num_cls` 的识别准确性）。
2. **如何选择 Issue？**
   - 优先选择标记为 `good first issue` 或 `help wanted` 的问题，适合新手或需要额外支持的贡献。
3. **我的 PR 被拒绝怎么办？**
   - 不要气馁！Maintainers 会提供反馈，您可以根据建议改进并重新提交。
4. **可以贡献非代码内容吗？**
   - 当然可以！文档改进、使用示例、翻译或问题报告都非常有价值。

## 联系我们

- **GitHub Issues**：提交 Bug、功能请求或问题：https://github.com/your-org/ccrs-library/issues
- **社区讨论**：参与 GitHub Discussions（如果启用）或联系 maintainers。

## 致谢

感谢所有为 **CCRS_Library** 贡献的开发者！您的努力帮助我们推动制造业智能质量检测的创新。让我们一起构建一个更强大的开源社区！
