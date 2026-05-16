# Usar uma imagem leve do Python
FROM python:3.11-slim

# Evitar que o Python gere arquivos .pyc e garantir que logs apareçam imediatamente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Definir o diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para algumas bibliotecas (se houver)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas o requirements.txt primeiro para aproveitar o cache das camadas do Docker
COPY requirements.txt .

# Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto
COPY . .

# Expor a porta que o Cloud Run utiliza (padrão 8080)
EXPOSE 8080

# Comando para rodar o Streamlit
# O Cloud Run passa a porta na variável de ambiente $PORT
CMD ["sh", "-c", "streamlit run 0_🧑🏻‍🔬_Home.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
