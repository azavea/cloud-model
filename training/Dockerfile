FROM quay.io/jmcclain/raster-vision-pytorch:Tue_Jun_29_05_00_22_UTC_2021

COPY catalogs.json /workdir/catalogs.json
COPY pipeline.py /workdir/pipeline.py
COPY default /root/.rastervision/default

WORKDIR /workdir

CMD ["bash"]
