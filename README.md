![The Clouds](https://upload.wikimedia.org/wikipedia/commons/4/46/Socrates_in_a_basket.jpg)

# Inference #

## Obtain Docker Image ##

Either build the docker image or pull it from quay.io.

### Build ###

(Note that the model bundles necessary to build this image are not currently checked into this repository.)

```bash
docker build -f Dockerfile.inference -t quay.io/jmcclain/cloud-model:latest .
```

### Pull from quay.io ###

```
docker pull quay.io/jmcclain/cloud-model:latest
```

## Perform Inference ##

```bash
docker run -it --rm \
       --runtime=nvidia --shm-size 16G \
       -v $HOME/Desktop/imagery:/input:ro \
       -v /tmp:/output \
       quay.io/jmcclain/cloud-model \
          --infile /input/greenwhich/L2A-0.tif \
          --outfile-final /output/final.tif \
          --outfile-raw /output/raw.tif \
          --level L2A \
          --architectures both
```

# Training #

## Build Docker Image ##

(Note that the file `catalog.json`, which is necessary for building the image and training, is not currently checked into this repository.)

```bash
docker build -t cloud-model -f Dockerfile .
```

## Run Container ##

```bash
docker run -it --rm \
       --name cloud-model --runtime=nvidia \
       --shm-size 16G \
       -v $HOME/.aws:/root/.aws:ro \
       cloud-model bash
```

## Invoke Raster-Vision ##

### Local ###

```bash
ROOT=/tmp/xxx ; \
rastervision run inprocess /workdir/pipeline.py \
       -a root_uri ${ROOT} \
       -a analyze_uri ${ROOT}/analyze \
       -a chip_uri ${ROOT}/chips \
       -a json catalogs.json \
       -a epochs 2 \
       -a batch_sz 2 \
       -a small_test True \
       chip train
```

### On AWS ###

#### Chip ####

```bash
LEVEL='L1C' ; \
ROOT="s3://bucket/prefix" ; \
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/0 \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -s 800 \
       chip
```

#### Train ####

```bash
LEVEL='L1C' ; \
ARCH=cheaplab ; \
ROOT="s3://bucket/prefix" ; \
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/${ARCH}-${LEVEL} \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -a architecture ${ARCH} \
       train
```
