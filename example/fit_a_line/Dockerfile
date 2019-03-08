FROM paddlepaddle/paddlecloud-job:0.11.0
RUN mkdir -p /data/recordio/imikolov && \
    python -c "import paddle; import paddle.v2.dataset as dataset; word_dict = dataset.imikolov.build_dict();  \
      dataset.imikolov.train(word_dict, 5); dataset.imikolov.test(word_dict, 5); \
      dataset.common.convert('/data/recordio/imikolov/', dataset.imikolov.train(word_dict, 5), 5000, 'imikolov-train')"

RUN mkdir -p /workspace
ADD train_ft.py /workspace
