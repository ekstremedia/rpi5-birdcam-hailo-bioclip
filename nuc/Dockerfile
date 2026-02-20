FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    open_clip_torch \
    flask \
    pillow

COPY species_api.py /app/species_api.py
WORKDIR /app

# Download model at build time so container starts fast
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip'); \
print('Model downloaded')"
# Clear token from image
ENV HF_TOKEN=""

EXPOSE 5555

CMD ["python", "species_api.py"]
