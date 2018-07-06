FROM yancey1989/paddleedl-job:0.11.0
RUN mkdir -p /data/mnist && \
    python -c "import paddle.v2.dataset as dataset; dataset.mnist.train(); dataset.mnist.test(); dataset.common.convert('/data/mnist', dataset.mnist.train(), 100, 'mnist-train')"
RUN mkdir -p /workspace
ADD train_ft.py /workspace

