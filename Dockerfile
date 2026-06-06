FROM 127.0.0.1:5000/zodiac/zodiac-python-base:1.0

# 覆盖默认端口
ENV UVICORN_PORT=16900

# 先复制依赖文件（利用 Docker 层缓存）
COPY ./application/pyproject.toml /app/
COPY ./application/uv.lock /app/
# 安装依赖（如果 pyproject.toml 或 uv.lock 没变，这一层会使用缓存）
RUN uv sync --frozen --no-dev
# 再复制所有源代码到工作目录
COPY ./application/. /app
# 设置容器启动入口
ENTRYPOINT ["/app/python-app-start.sh"]