This custom node for ComfyUI integrates a quantized version of the Molmo-7B-D model, allowing users to generate detailed image captions and analyses directly within their ComfyUI workflows.

## Features

- Image captioning using a 4-bit quantized version of Molmo-7B-D
- Support for both general description and detailed analysis
- Custom prompt input option
- Adjustable generation parameters (max tokens, temperature, top_k, top_p)
- Automatic dependency installation

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:

   ```
   git clone https://github.com/CY-CHENYUE/ComfyUI-Molmo.git
   ```

2. Restart ComfyUI. The node will automatically install its dependencies on first use.

## Usage

After installation, you can find the "Molmo 7B D bnb 4bit" node in the "Molmo" category in ComfyUI's node menu.

### Input Parameters

- `image`: The input image to be captioned or analyzed
- `prompt_type`: Choose between "Describe" for general captioning or "Detailed Analysis" for a more comprehensive breakdown
- `custom_prompt`: Optional. If provided, overrides the selected prompt type
- `seed`: Seed for reproducibility (0 for random)
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness in generation
- `top_k`: Limits vocabulary for next-word selection
- `top_p`: Nucleus sampling parameter

### Output

- `STRING`: The generated caption or analysis

## Notes

- The model is automatically downloaded on first use if not already present
- Requires a CUDA-capable GPU for optimal performance
- Initial load time may be longer due to model size

## Performance

The original Molmo 7B-D model, which this quantized version is based on, has shown impressive performance:

- Average Score on 11 Academic Benchmarks: 77.3
- Human Preference Elo Rating: 1056

These scores place it competitively among other large language models, including some versions of GPT-4 and Claude. For more detailed comparisons, please refer to the [original model's page](https://huggingface.co/allenai/Molmo-7B-D-0924).

## Acknowledgements

- [Original Molmo-7B-D model](https://huggingface.co/allenai/Molmo-7B-D-0924) by Allen Institute for AI
- [Quantized Molmo-7B-D-bnb-4bit model](https://huggingface.co/cyan2k/molmo-7B-D-bnb-4bit) by cyan2k
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.