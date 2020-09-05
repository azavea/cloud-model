# FROM raster-vision-pytorch:latest
FROM quay.io/azavea/raster-vision:pytorch-7abe1e8

RUN pip3 install --upgrade pystac==0.5.2

COPY pipeline.py /workdir/pipeline.py
COPY default /root/.rastervision/default

CMD ["bash"]
