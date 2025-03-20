# Use a base image with Python3.9 and Nodejs20 slim version
FROM nikolaik/python-nodejs:python3.9-nodejs20-slim as static

WORKDIR /app


RUN apt update &&\
    apt install -y wget


RUN rm -rf static && \
    wget -O dist.tar.gz https://public-frontend-1300249583.cos.ap-nanjing.myqcloud.com/test-hp-metagpt-web/dist-20240417204906.tar.gz &&\
    tar xvzf dist.tar.gz && \
    mv dist static && \
    rm dist.tar.gz


FROM nikolaik/python-nodejs:python3.9-nodejs20-slim

USER root

# Install Debian software needed by MetaGPT and clean up in one RUN command to reduce image size
RUN apt update &&\
    apt install -y git chromium fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 --no-install-recommends &&\
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI globally
ENV CHROME_BIN="/usr/bin/chromium" \
    PUPPETEER_CONFIG="/app/config/puppeteer-config.json"\
    PUPPETEER_SKIP_CHROMIUM_DOWNLOAD="true"

RUN npm install -g @mermaid-js/mermaid-cli &&\
    npm cache clean --force

WORKDIR /app

# Install Python dependencies and install MetaGPT
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /app/logs && chmod 777 /app/logs && \
    mkdir -p /app/workspace && chmod 777 /app/workspace && \
    mkdir -p /app/storage && chmod 777 /app/storage

COPY . .
COPY --from=static /app/static /app/static
COPY config/template.yaml static/config.yaml

CMD ["python", "app.py"]
