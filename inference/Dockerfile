FROM quay.io/jmcclain/raster-vision-pytorch:Thu_Dec_10_05_00_51_UTC_2020 

# Models
COPY models/ /workdir/models/

# Prediction Script
COPY predict.py /workdir/

# Load pretrained models
RUN /workdir/predict.py --level L1C --exit-early True --device cpu --infile xxx --outfile-final xxx --outfile-raw xxx
RUN /workdir/predict.py --level L2A --exit-early True --device cpu --infile xxx --outfile-final xxx --outfile-raw xxx

WORKDIR /workdir

ENTRYPOINT ["/workdir/predict.py"]
