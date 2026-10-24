FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY sdd ./sdd
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e . streamlit
COPY app ./app
COPY configs ./configs
COPY examples ./examples
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
