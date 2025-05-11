import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests # For downloading image from URL
import numpy as np
import cv2 # OpenCV for image manipulation
import matplotlib.pyplot as plt
import timm # PyTorch Image Models

class GradCAM:
    """
    PyTorch Grad-CAM implementation.
    Helps in visualizing the regions of input image that were important for a an image classification model's prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _capture_gradients(self, module, grad_input, grad_output):
        """Hook for capturing gradients."""
        self.gradients = grad_output[0].detach() # grad_output is a tuple

    def _capture_activations(self, module, input, output):
        """Hook for capturing activations."""
        self.activations = output.detach()

    def _register_hooks(self):
        """Registers forward and backward hooks to the target layer."""
        if self.target_layer is None:
            raise ValueError("Target layer cannot be None.")
        
        # Backward hook to get gradients
        self.target_layer.register_full_backward_hook(self._capture_gradients)
        # Forward hook to get activations
        self.target_layer.register_forward_hook(self._capture_activations)
        print(f"Hooks registered on layer: {self.target_layer}")

    def generate_heatmap(self, input_tensor, class_idx=None, retain_graph=False):
        """
        Generates the Grad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor.
            class_idx (int, optional): Index of the target class. If None, the class with the highest score is used.
            retain_graph (bool): Whether to retain the computation graph after backward pass. Set to True if you need to backpropagate multiple times.

        Returns:
            numpy.ndarray: The generated heatmap.
        """
        self.model.eval() # Set model to evaluation mode
        self.model.zero_grad() # Zero out any existing gradients

        # Forward pass to get model output
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Target for backpropagation: score of the target class
        target_score = output[:, class_idx]

        # Backward pass to compute gradients
        target_score.backward(retain_graph=retain_graph)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations. Check hook registration and model structure.")

        # Get gradients and activations
        gradients = self.gradients # Shape: (batch_size, num_channels, H, W)
        activations = self.activations # Shape: (batch_size, num_channels, H, W)

        # Pool the gradients across the spatial dimensions (H, W) to get neuron importance weights
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # Shape: (num_channels,)

        # Weight the channels in the activation maps
        # For each channel, multiply the activation map by its corresponding gradient weight
        for i in range(activations.shape[1]): # Iterate over channels
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the channels of the weighted activation maps to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze() # Squeeze to remove batch and channel dim if they are 1
        
        # Apply ReLU to keep only positive contributions
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

def preprocess_image(img_path, image_size=(300, 300)):
    """
    Loads and preprocesses an image for PyTorch models.
    Args:
        img_path (str): Path to the image or URL.
        image_size (tuple): Target size (height, width) for the image.
    Returns:
        tuple: (PIL.Image, torch.Tensor) Original PIL image and preprocessed image tensor.
    """
    if img_path.startswith('http'):
        try:
            response = requests.get(img_path, stream=True)
            response.raise_for_status()
            img_pil = Image.open(response.raw).convert('RGB')
        except requests.exceptions.RequestException as e:
            print(f"下载图像时出错: {e}")
            return None, None
    else:
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 图像文件 {img_path} 未找到。")
            return None, None
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None, None

    # Standard ImageNet normalization. 
    # 注意: 如果您的模型在训练时使用了不同的预处理方法，您可能需要调整这里的转换。
    preprocess_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess_transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension: (C, H, W) -> (1, C, H, W)
    return img_pil, img_tensor

def display_gradcam_pytorch(original_pil_img, heatmap_numpy, model_name_for_title="Model", alpha=0.5, superimposed_img_path="superimposed_gradcam_pytorch.jpg"):
    """
    Displays the original image, heatmap, and superimposed image.
    Args:
        original_pil_img (PIL.Image): The original image.
        heatmap_numpy (numpy.ndarray): The Grad-CAM heatmap.
        model_name_for_title (str): Name of the model for plot titles.
        alpha (float): Transparency for heatmap overlay.
        superimposed_img_path (str): Path to save the superimposed image.
    """
    img_cv = cv2.cvtColor(np.array(original_pil_img), cv2.COLOR_RGB2BGR) # PIL RGB to OpenCV BGR

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap_numpy, (img_cv.shape[1], img_cv.shape[0]))
    
    # Apply colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = heatmap_colored * alpha + img_cv * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    cv2.imwrite(superimposed_img_path, superimposed_img)
    print(f"Grad-CAM 叠加图像已保存至: {superimposed_img_path}")

    # Display using Matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_pil_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title(f"Grad-CAM heatmap ({model_name_for_title})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # Convert superimposed_img from BGR (OpenCV) to RGB (Matplotlib)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM mix ({model_name_for_title})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Removed get_imagenet_labels function as it's not needed for a custom 3-class model.

def main():
    # --- Configuration ---
    model_name = 'efficientnet_b3' # 您使用的模型架构
    num_custom_classes = 3 # 您的模型分类数量

    # 定义您的类别名称
    custom_class_names = {
        0: "Real Face",
        1: "Clean Fake Face",
        2: "Noisy Fake Face"
    }
    
    # 根据模型名称设置图像尺寸
    # EfficientNetB3 的标准输入尺寸是 300x300
    if model_name == 'efficientnet_b0':
        img_size = (224, 224)
    elif model_name == 'efficientnet_b1':
        img_size = (240, 240)
    elif model_name == 'efficientnet_b2':
        img_size = (260, 260)
    elif model_name == 'efficientnet_b3':
        img_size = (300, 300)
    elif model_name == 'efficientnet_b4':
        img_size = (380, 380)
    elif model_name == 'efficientnet_b5':
        img_size = (456, 456)
    elif model_name == 'efficientnet_b6':
        img_size = (528, 528)
    elif model_name == 'efficientnet_b7':
        img_size = (600, 600)
    else: 
        img_size = (224, 224) 
        print(f"警告: 模型 {model_name} 的输入尺寸未知, 使用默认尺寸 {img_size}. 请验证并修改 img_size。")

    print(f"使用模型: {model_name} (自定义 {num_custom_classes} 分类)，输入图像尺寸: {img_size}")

    img_path = '/root/autodl-tmp/DFGC_Detection-master/data_preparation/data_structure/Celeb-DF-v2-face/Celeb-synthesis_adv1/id0_id1_0000.mp4/0274.png' # 示例图片，请替换为您的测试图片
    # img_path = 'your_local_face_image.jpg' # 例如，您的本地人脸图片路径

    # 1. Load model structure and then your custom trained weights
    try:
        # 加载模型结构，不加载预训练的 ImageNet 权重，并指定分类类别数
        model = timm.create_model(model_name, pretrained=False, num_classes=num_custom_classes) 
        
        # !!! 重要: 请将下面的路径替换为您训练好的模型权重文件的实际路径 !!!
        custom_weights_path = "/root/autodl-tmp/DeepfakesAdvTrack-Spring2025/detection/utils/84_acc0.9976base.pth" 
        # 例如: custom_weights_path = "/path/to/your/efficientnet_b3_3class_weights.pth"
        print(f"尝试从 '{custom_weights_path}' 加载自定义权重...")
        from utils.efficientnet import TransferModel
        model = TransferModel('efficientnet-b3', num_out_classes=3)
        model.load_state_dict(torch.load(custom_weights_path, map_location='cpu'), strict=False)
        

        
        model.eval() # Set to evaluation mode
        print(f"成功加载自定义训练权重: {custom_weights_path}")
    except FileNotFoundError:
        print(f"错误：找不到自定义权重文件 '{custom_weights_path}'。")
        print("请确保路径正确，并且文件存在。")
        print("Grad-CAM 将使用随机初始化的模型，结果可能没有意义。")
        # 如果权重加载失败，可以选择退出或继续（但结果可能无用）
        # return 
    except Exception as e:
        print(f"加载模型或自定义权重时出错: {e}")
        return

    # 2. Identify the target layer
    target_layer = None
    if hasattr(model, 'conv_head'): 
        target_layer = model.conv_head
        print(f"使用目标层: model.conv_head")
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        if hasattr(last_block, 'conv_pwl'): 
            target_layer = last_block.conv_pwl
            print(f"从最后一个块中选择目标层: conv_pwl")
        elif hasattr(last_block, 'conv_dw'): 
             target_layer = last_block.conv_dw
             print(f"从最后一个块中选择目标层: conv_dw")
        elif hasattr(last_block, 'conv3'): 
             target_layer = last_block.conv3
             print(f"从最后一个块中选择目标层: conv3")
        elif hasattr(last_block, 'conv2'):
             target_layer = last_block.conv2
             print(f"从最后一个块中选择目标层: conv2")
        elif hasattr(last_block, 'conv1'):
             target_layer = last_block.conv1
             print(f"从最后一个块中选择目标层: conv1")

    if target_layer is None:
        print("未能通过常见名称找到目标层，尝试遍历查找最后一个 Conv2d...")
        candidate_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                is_classifier_conv = 'classifier' in name or 'fc' in name or 'head' in name and name != 'conv_head'
                if not is_classifier_conv:
                     candidate_layers.append(module)
        if candidate_layers:
            target_layer = candidate_layers[-1] 
            print(f"使用后备目标层 (候选中的最后一个 Conv2d): {target_layer}")
        else:
            print("错误: 无法自动确定合适的目标卷积层。")
            print("请检查模型结构 (例如 print(model) 或 list(model.named_modules())) 并手动设置 target_layer。")
            return
            
    # 3. Initialize GradCAM
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # 4. Load and preprocess image
    original_pil_img, input_tensor = preprocess_image(img_path, image_size=img_size)
    if input_tensor is None:
        print(f"无法处理图像: {img_path}")
        return

    # 5. Make prediction
    with torch.no_grad(): 
        output = model(input_tensor)
    
    probabilities = F.softmax(output, dim=1)[0]
    predicted_class_idx = torch.argmax(probabilities).item()

    # 使用自定义类别名称
    if predicted_class_idx in custom_class_names:
        predicted_class_name = custom_class_names[predicted_class_idx]
    else:
        predicted_class_name = f"未知类别 (Unknown Class {predicted_class_idx})"

    print(f"预测类别: {predicted_class_name} (索引: {predicted_class_idx}), 置信度: {probabilities[predicted_class_idx].item():.4f}")
    
    print(f"所有类别置信度:")
    for i in range(num_custom_classes):
        class_name = custom_class_names.get(i, f"未知类别 {i}")
        print(f"  {class_name}: {probabilities[i].item():.3f}")


    # 6. Generate Grad-CAM heatmap
    print(f"为类别 '{predicted_class_name}' (索引: {predicted_class_idx}) 生成 Grad-CAM")
    heatmap_numpy = grad_cam.generate_heatmap(input_tensor, class_idx=predicted_class_idx)

    # 7. Display results
    safe_class_name = "".join(c if c.isalnum() else "_" for c in predicted_class_name.split('(')[0].strip()) # 使用类别名称创建安全的文件名
    output_filename = f"superimposed_gradcam_{model_name}_{safe_class_name}_class{predicted_class_idx}.jpg"
    display_gradcam_pytorch(original_pil_img, heatmap_numpy, model_name_for_title=f"{model_name} - {predicted_class_name}", superimposed_img_path=output_filename)

if __name__ == '__main__':
    main()
