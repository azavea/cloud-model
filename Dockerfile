FROM quay.io/azavea/raster-vision:pytorch-1d37fe8

RUN pip3 install --upgrade pystac==0.5.2 && apt-get install -y nano

COPY pipeline.py /workdir/pipeline.py
COPY default /root/.rastervision/default

CMD ["bash"]
