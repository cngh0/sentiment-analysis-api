# ---- Stage 1: Builder ----
# 使用一个包含完整编译工具的Python镜像作为“构建环境”
FROM python:3.9 as builder

WORKDIR /usr/src/app

# 先安装依赖，利用Docker缓存
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt

# ---- Stage 2: Final Image ----
# 使用一个极度轻量的slim镜像作为“运行环境”，减小最终镜像体积
FROM python:3.9-slim

WORKDIR /app

# 从构建环境中复制已编译好的依赖包
COPY --from=builder /usr/src/app/wheels /wheels
RUN pip install --no-cache /wheels/*

# 复制我们的应用代码和模型文件
COPY ./app ./app
COPY ./models ./models

# 暴露端口
EXPOSE 5000

# 使用gunicorn启动应用，这是生产环境的标准做法
# --workers 3: 启动3个工作进程来处理请求
# --bind 0.0.0.0:5000: 监听所有网络接口的5000端口
# app.app:app: 从app/app.py文件中找到名为app的Flask实例
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5000", "app.app:app"]