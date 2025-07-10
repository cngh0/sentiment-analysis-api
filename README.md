# 情感分析API服务与容器化部署

这是一个端到端的机器学习项目，旨在将一个情感分析模型封装成一个可通过网络访问的、标准化的RESTful API服务，并使用Docker进行容器化部署。

## ✨ 项目亮点

- **双模型对比**: 同时集成了基于`Scikit-learn`的传统模型(V1)和基于`Hugging Face Transformers (DistilBERT)`的深度学习模型(V2)，可直观对比二者在复杂语境下的性能差异。
- **工程化实践**: 完整覆盖了从模型开发、API封装到生产级部署的MLOps流程。
- **容器化**: 采用**多阶段Dockerfile**构建轻量、可移植的Docker镜像，实现了“一次构建，处处运行”。
- **生产级部署**: 在容器中使用`Gunicorn`作为WSGI服务器，确保服务的稳定性和高并发能力。

## 🛠️ 技术栈

- **Python**: 主要编程语言
- **机器学习**: Scikit-learn, Hugging Face Transformers, PyTorch (CPU)
- **Web开发**: Flask, Gunicorn
- **容器化**: Docker
- **核心工具**: Git, WSL2

## 🚀 如何运行

### 1. 环境准备

- 确保你的电脑已安装 Docker。
- 克隆本项目: `git clone [你的GitHub仓库链接]`
- 进入项目目录: `cd sentiment_api_project`

### 2. 构建并运行Docker容器

```bash
# 构建镜像 (我们将版本号标记为2.0)
docker build -t sentiment-api:2.0 .

# 运行容器，并将主机的5001端口映射到容器的5000端口
docker run -p 5001:5000 --rm sentiment-api:2.0
```

### 3. 测试API

服务启动后，你可以使用`curl`或其他API工具进行测试。

**测试V1 (Scikit-learn)模型:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "The acting was great, but the plot was a complete waste of time."}' \
http://127.0.0.1:5001/predict
```

**测试V2 (DistilBERT)模型:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "The acting was great, but the plot was a complete waste of time."}' \
http://127.0.0.1:5001/predict_v2
```

## 📝 API端点说明

- `POST /predict`: 使用Scikit-learn模型进行预测。
- `POST /predict_v2`: 使用DistilBERT模型进行预测，能更好地处理复杂语境。

**请求体格式:**
```json
{
  "text": "Your text to analyze here."
}
