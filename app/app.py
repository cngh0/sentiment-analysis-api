# 文件路径: app/app.py
from flask import Flask, request, jsonify
import joblib
import os
# --- 导入必要的Hugging Face和PyTorch库 ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化Flask应用
app = Flask(__name__)

# ====================================================================
# V1 模型加载 (Scikit-learn) - 保持不变
# ====================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment_vectorizer.joblib')

try:
    model_v1 = joblib.load(MODEL_PATH)
    vectorizer_v1 = joblib.load(VECTORIZER_PATH)
    print("V1模型(Scikit-learn)加载成功。")
except Exception as e:
    print(f"加载V1模型失败: {e}")
    model_v1 = None
    vectorizer_v1 = None

# ====================================================================
# V2 模型加载 (Hugging Face) - 手动加载，深度掌控
# ====================================================================
try:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # 加载分词器 (Tokenizer)
    tokenizer_v2 = AutoTokenizer.from_pretrained(model_name)
    # 加载模型
    model_v2 = AutoModelForSequenceClassification.from_pretrained(model_name)
    # 明确设置模型在CPU上运行，并进入评估模式
    model_v2.to('cpu')
    model_v2.eval()
    print("V2模型(Tokenizer+Model)在CPU上加载成功。")
except Exception as e:
    print(f"加载V2模型失败: {e}")
    model_v2 = None
    tokenizer_v2 = None

# ====================================================================
# API 端点
# ====================================================================

# V1 预测端点 - 保持不变
@app.route('/predict', methods=['POST'])
def predict_v1():
    if not model_v1 or not vectorizer_v1:
        return jsonify({'error': 'V1模型未加载，无法进行预测。'}), 500

    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({'error': "请求中缺少'text'字段。"}), 400

    text = data['text']
    vectorized_text = vectorizer_v1.transform([text])
    prediction = model_v1.predict(vectorized_text)
    
    return jsonify({'version': 'v1_sklearn', 'text': text, 'sentiment': prediction[0]})

# V2 预测端点 - 这是修正后的版本
@app.route('/predict_v2', methods=['POST'])
def predict_v2():
    if not model_v2 or not tokenizer_v2:
        return jsonify({'error': 'V2模型未加载，无法进行预测。'}), 500

    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({'error': "请求中缺少'text'字段。"}), 400
        
    text = data['text']
    # 增加一个健壮性检查，确保输入是字符串
    if not isinstance(text, str):
        return jsonify({'error': "输入'text'必须是单个字符串。"}), 400

    try:
        # 1. 使用分词器对文本进行编码
        inputs = tokenizer_v2(text, return_tensors="pt", truncation=True, padding=True)
        inputs.to('cpu')

        # 2. 模型进行推理，并关闭梯度计算以节省资源
        with torch.no_grad():
            outputs = model_v2(**inputs)
            logits = outputs.logits

        # 3. 解码结果
        # 从logits中找到概率最高的类别的ID
        predicted_class_id = logits.argmax().item()
        # 使用模型的配置将ID转换为可读的标签 (e.g., 0 -> 'NEGATIVE', 1 -> 'POSITIVE')
        sentiment = model_v2.config.id2label[predicted_class_id]
        
        return jsonify({'version': 'v2_distilbert_manual', 'text': text, 'sentiment': sentiment})

    except Exception as e:
        # 捕获潜在的运行时错误
        print(f"V2预测时发生错误: {e}")
        return jsonify({'error': '预测过程中发生内部错误。'}), 500

# --- 新增健康检查端点 ---
@app.route('/health', methods=['GET'])
def health_check():
    # 这个端点不执行复杂操作，只用于让Render知道服务是正常的
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    

# curl -X POST -H "Content-Type: application/json" -d '{"text": "The acting was great, but the plot was a complete waste of time."}' http://127.0.0.1:5000/predict
# curl -X POST -H "Content-Type: application/json" -d '{"text": "The acting was great, but the plot was a complete waste of time."}' http://127.0.0.1:5000/predict_v2