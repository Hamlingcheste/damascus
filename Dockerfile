FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /work

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["damascus"]
CMD ["--help"]
