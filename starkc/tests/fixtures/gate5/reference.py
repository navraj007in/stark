import sys
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # Resize shorter side to 256
    if w < h:
        new_w = 256
        new_h = int(round(h * 256.0 / w))
    else:
        new_h = 256
        new_w = int(round(w * 256.0 / h))
        
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Center crop 224x224
    x0 = (new_w - 224) // 2
    y0 = (new_h - 224) // 2
    img = img.crop((x0, y0, x0 + 224, y0 + 224))
    
    # Scale from [0, 255] to [0, 1] range
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    
    # Normalization (identical to STARK: channel-wise subtraction/division)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    arr = (arr - mean) / std
    return arr

def main():
    if len(sys.argv) < 3:
        print("Usage: reference.py <model.onnx> <image.jpg>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    input_tensor = preprocess(image_path)
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0][0]
    
    # Stable Softmax
    max_logit = np.max(logits)
    exp = np.exp(logits - max_logit)
    probs = exp / np.sum(exp)
    
    # Load labels
    labels = []
    labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(model_path))), "tmp", "imagenet_classes.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f]
            
    # Sort top 5 descending
    top_indices = np.argsort(probs)[::-1][:5]
    
    print("top-1 class : {}".format(top_indices[0]))
    print("probability : {:.6f}".format(probs[top_indices[0]]))
    print("logits sum : {:.6f}".format(np.sum(logits)))
    print("probs sum : {:.6f}".format(np.sum(probs)))
    
    print("\nTop 5 predictions:")
    for rank, idx in enumerate(top_indices, 1):
        label = labels[idx] if idx < len(labels) else "unknown"
        print(f"  {rank}. Class {idx} ({label}): logit={logits[idx]:.4f}, prob={probs[idx]:.6f}")

if __name__ == '__main__':
    main()
