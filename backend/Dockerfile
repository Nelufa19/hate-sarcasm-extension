# Hugging Face Docker Spaces run containers as UID 1000, not root -- create
# that user and own everything under it, or every COPY/RUN after this point
# will hit permission errors when the Space actually boots the container.
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Model checkpoints (~770MB total) are copied in directly rather than
# downloaded at runtime -- HF Spaces has no persistent disk on the free
# tier's ephemeral filesystem across rebuilds, and baking them into the
# image means the Space works immediately with no external dependency.
COPY --chown=user models/ models/
COPY --chown=user app/ app/

# HF Spaces expects the app to listen on 7860 (see README.md's app_port).
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
