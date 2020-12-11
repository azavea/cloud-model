FROM quay.io/jmcclain/raster-vision-pytorch:Thu_Dec_10_05_00_51_UTC_2020

COPY catalogs.json /workdir/catalogs.json
COPY pipeline.py /workdir/pipeline.py
COPY default /root/.rastervision/default

WORKDIR /workdir

CMD ["bash"]
