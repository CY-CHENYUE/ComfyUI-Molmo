import os
import sys
import importlib.util
import random
import json

class Molmo7BDbnb:
    def __init__(self):
        self.device = None
        self.repo_name = "cyan2k/molmo-7B-D-bnb-4bit"
        self.model_path = None
        self.arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
        self.processor = None
        self.model = None
        self.dependencies_installed = self.load_installation_status()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt_type": (["Describe", "Detailed Analysis"],),
                "custom_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "Enter custom prompt here. If provided, this will override the prompt type selection."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "max_new_tokens": ("INT", {"default": 350, "min": 1, "max": 1000}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "Molmo"

    @classmethod
    def IS_CHANGED(s, image, prompt_type, custom_prompt, seed, max_new_tokens, temperature, top_k, top_p):
        if seed == 0:
            return float("nan")  # Always re-run
        return seed  # Re-run when seed changes

    def load_installation_status(self):
        status_file = os.path.join(os.path.dirname(__file__), "molmo_installation_status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                return status.get('installed', False)
        return False

    def save_installation_status(self):
        status_file = os.path.join(os.path.dirname(__file__), "molmo_installation_status.json")
        with open(status_file, 'w') as f:
            json.dump({'installed': True}, f)

    def load_dependencies(self):
        try:
            # 获取当前文件的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 将当前目录添加到 sys.path
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # 动态导入 install 模块
            import install
            
            # 保存当前工作目录
            original_cwd = os.getcwd()
            
            try:
                # 切换到节点目录
                os.chdir(current_dir)
                
                # 调用 install_dependencies 函数
                if not self.dependencies_installed:
                    install.install_dependencies()
                    self.save_installation_status()
                    self.dependencies_installed = True
                    raise Exception("Molmo dependencies have been installed. Please restart ComfyUI for the changes to take effect.")
                
                global torch, AutoModelForCausalLM, AutoProcessor, GenerationConfig, folder_paths, Image, ImageStat, np, snapshot_download
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
                import folder_paths
                from PIL import Image, ImageStat
                import numpy as np
                from huggingface_hub import snapshot_download

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model_path = os.path.join(folder_paths.models_dir, "Molmo", "molmo-7B-D-bnb-4bit")
            finally:
                # 恢复原来的工作目录
                os.chdir(original_cwd)

        except Exception as e:
            raise ImportError(f"Failed to load dependencies: {str(e)}")

    def load_model(self):
        if self.processor is None or self.model is None:
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                print(f"Model not found locally. Downloading to {self.model_path}")
                snapshot_download(repo_id=self.repo_name, local_dir=self.model_path)
            else:
                print(f"Model found locally at {self.model_path}")

            print(f"Loading model from {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(self.model_path, **self.arguments)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.arguments)

    def preprocess_image(self, image):
        # 将 numpy 数组转换为 PIL Image
        pil_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')
        
        # 计算平均亮度
        gray_image = pil_image.convert('L')
        stat = ImageStat.Stat(gray_image)
        average_brightness = stat.mean[0]

        # 根据亮度定义背景颜色
        bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

        # 创建带有背景色的新图像
        new_image = Image.new('RGB', pil_image.size, bg_color)
        new_image.paste(pil_image, (0, 0), pil_image if pil_image.mode == 'RGBA' else None)

        return new_image

    def generate_caption(self, image, prompt_type, custom_prompt, seed, max_new_tokens, temperature, top_k, top_p):
        # 选择 prompt
        prompts = {
            "Describe": "Describe this image in detail.",
            "Detailed Analysis": "Analyze this image, including its style, theme, scene, composition, lighting, and any additional notable information."
        }
        
        # 如果用户有输入，就用用户的输入
        if custom_prompt.strip():
            selected_prompt = custom_prompt
            prompt_note = "\n[Note: Custom prompt was used, overriding the selected prompt type.]"
        else:
            selected_prompt = prompts[prompt_type]
            prompt_note = ""

        # 延迟加载依赖和模型
        if self.device is None:
            self.load_dependencies()
        if self.model is None:
            self.load_model()

        if seed == 0:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = seed % (2**32)  # 确保种子在有效范围内
        
        # 设置全局随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保 CuDNN 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        processed_image = self.preprocess_image(image)
        
        inputs = self.processor.process(
            images=[processed_image],
            text=selected_prompt
        )
        
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # 设置生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings="<|endoftext|>",
            do_sample=True,  # 启用采样
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 使用 torch.Generator 设置随机状态
        with torch.random.fork_rng(devices=[self.model.device]):
            torch.random.manual_seed(seed)
            
            output = self.model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=self.processor.tokenizer
            )
        
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        caption = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 添加提示注释
        caption += prompt_note
        
        return (caption,)

NODE_CLASS_MAPPINGS = {
    "Molmo7BDbnb": Molmo7BDbnb
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Molmo7BDbnb": "Molmo 7B D bnb 4bit"
}
