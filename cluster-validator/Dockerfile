FROM nvcr.io/nvidia/pytorch:25.01-py3
# Base image includes CUDA, PyTorch, NCCL, MPI, and nccl-tests at /usr/local/bin/.

WORKDIR /validator

RUN apt-get update -q && \
    apt-get install -y -q --no-install-recommends \
        iperf3 ibverbs-utils infiniband-diags bc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY checks/ checks/
COPY report/  report/
COPY run.sh   run.sh

RUN chmod +x run.sh checks/*.sh

ENV SCRATCH_DIR=/tmp/validator_results

ENTRYPOINT ["./run.sh"]
