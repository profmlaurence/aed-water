# Usa uma imagem oficial do Python leve
FROM python:3.11-slim

# Evita que o Python grave arquivos .pyc e força o log no console
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala o Chromium e bibliotecas de sistema necessárias para o Kaleido/Chromium rodar em modo headless
RUN apt-get update && apt-get install -y \
    chromium \
    libnss3 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Força o Kaleido a encontrar o Chromium
ENV BROWSER_PATH=/usr/bin/chromium

# Copia os arquivos de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Expõe a porta que o Cloud Run espera (8080 por padrão)
EXPOSE 8080

# O Cloud Run passa a porta recomendada via variável de ambiente PORT.
# Usamos o formato shell (sem os colchetes []) para que a variável $PORT seja interpretada corretamente.
CMD streamlit run "0_🧑🏻‍🔬_Home.py" --server.port=${PORT:-8080} --server.address=0.0.0.0