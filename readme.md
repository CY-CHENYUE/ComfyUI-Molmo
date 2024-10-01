# ComfyUI-Molmo

使用molmo模型，在ComfyUI中实现图片描述，分析图片内容。可以把图片转文本的结果作为提示词生成图片。

## 功能

- 图像转文本
- 支持一般描述和详细分析
- 自定义提示输入选项
- 可调节的生成参数(max tokens, temperature, top_k, top_p)
- 图像到文本转换,可用于生成提示词

## 安装

1. 在ComfyUI的管理器中搜索并安装"ComfyUI-Molmo"。

2. 或者，也可以手动克隆此仓库到ComfyUI的`custom_nodes`目录:

   ```
   git clone https://github.com/CY-CHENYUE/ComfyUI-Molmo.git
   ```

3. 重启ComfyUI。

4. 依赖安装：
   - 首次运行节点时，将自动下载并安装所需的依赖项。
   - 注意：部分依赖可能需要重启ComfyUI后才能生效。如果遇到问题，请尝试重新启动ComfyUI。

5. 模型下载：
   - 如果模型文件不存在，将在首次使用时自动下载。
   - 由于模型文件较大，下载可能需要一些时间，请耐心等待。

注意：初次使用时，由于需要下载模型和安装依赖，启动时间可能会较长。后续使用将会更快。

## 使用

安装后,您可以在ComfyUI的节点菜单的"Molmo"类别中找到"Molmo 7B D bnb 4bit"节点。

### 输入参数

- `image`: 要描述或分析的输入图像
- `prompt_type`: 选择"Describe"进行一般描述或"Detailed Analysis"进行更全面的分析
- `custom_prompt`: 可选。如果提供,将覆盖选定的prompt_type
- `seed`: 用于可重现性的种子
- `max_new_tokens`: 生成的最大标记数
- `temperature`: 控制生成的随机性
- `top_k`: 限制下一个词选择的词汇表
- `top_p`: 核采样参数

### 输出

- `STRING`: 生成的描述或分析

## 示例
<video controls src="images/molmo-v.mp4" title="Title"></video>

![alt text](images/MOLMO-CY-V1.png)

## 注意事项

- 如果模型不存在,将在首次使用时自动下载。国内环境可以通过网盘下载 https://pan.quark.cn/s/882aea39f59c 后放到ComfyUI/models。
- 需要CUDA兼容的GPU以获得最佳性能
- 由于模型大小,初始加载时间可能较长

## 性能

模型使用基于Molmo 7B-D的量化版本，可以降低显存的使用。

## 致谢

- Allen Institute for AI的[原始Molmo-7B-D模型](https://huggingface.co/allenai/Molmo-7B-D-0924)
- cyan2k的[量化Molmo-7B-D-bnb-4bit模型](https://huggingface.co/cyan2k/molmo-7B-D-bnb-4bit)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)项目

## 贡献

欢迎贡献!请随时提交Pull Request.