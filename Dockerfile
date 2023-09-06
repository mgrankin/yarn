FROM nvcr.io/nvidia/pytorch:23.08-py3
USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN yes | apt install libpq-dev libaio-dev

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary

COPY . /opt/app
RUN pip install -e .

EXPOSE 8000/tcp
CMD cd eval; uvicorn serve:app --host 0.0.0.0 --port 8000