#!/bin/bash
# download_test_model.sh

MODEL_DIR="$HOME/.codeatlas/models"
mkdir -p "$MODEL_DIR"

echo "选择要下载的模型:"
echo "1) Qwen2.5-Coder-1.5B (推荐, ~1.5GB)"
echo "2) Llama-3.2-1B (最小, ~700MB)"
echo "3) Phi-3.5-mini (平衡, ~2.5GB)"
echo "4) Qwen2.5-Coder-7B (~4GB)"
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        FILE="qwen2.5-coder-1.5b.gguf"
        ;;
    2)
        URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        FILE="llama-3.2-1b.gguf"
        ;;
    3)
        URL="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        FILE="phi-3.5-mini.gguf"
        ;;
    4)
        URL="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        FILE="qwen2.5-coder-7b.gguf"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo "下载中: $FILE"
curl -L -o "$MODEL_DIR/$FILE" "$URL"
echo "完成! 模型保存到: $MODEL_DIR/$FILE"
