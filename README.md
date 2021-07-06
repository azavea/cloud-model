![The Clouds](https://upload.wikimedia.org/wikipedia/commons/4/46/Socrates_in_a_basket.jpg)

# Inference #

## Obtain Docker Image ##

Either build the docker image or pull it from quay.io.

### Build ###

Note that the model bundles necessary to build this image are not currently checked into this repository.
The models can be obtained by typing the following.
```bash
cd inference/
aws s3 sync s3://azavea-cloud-model/models models --request-payer requester --dryrun
```

The docker image can be built by typing the following (with or without the change of directory).
```bash
cd inference/
docker build -f Dockerfile -t quay.io/jmcclain/cloud-model:latest .
```

### Pull from quay.io ###

```
docker pull quay.io/jmcclain/cloud-model:latest
```

## Perform Inference ##

```bash
cd inference/
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

```bash
cd training/
docker build -t azavea-cloud-model-training -f Dockerfile .
```

## Run Container ##

```bash
cd training/
docker run -it --rm \
       --name azavea-cloud-model-training --runtime=nvidia \
       --shm-size 16G \
       -v $HOME/.aws:/root/.aws:ro \
       azavea-cloud-model-training bash
```

## Invoke Raster-Vision ##

### Local ###

```bash
export AWS_REQUEST_PAYER=requester
export ROOT=/tmp/xxx
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

It is required to have a compute environment with [additional storage](https://aws.amazon.com/premiumsupport/knowledge-center/batch-ebs-volumes-launch-template/) for the `p3.2xlarge` batch instance that is used for training.  (The large number of chips will not fit on a volume of the default size.)

#### Chip ####

```bash
export AWS_REQUEST_PAYER='requester'
export LEVEL='L1C'
export ROOT='s3://bucket/prefix'
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/xxx \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -s 800 \
       chip
```

#### Train ####

```bash
export LEVEL='L1C'
export ARCH='cheaplab'
export ROOT='s3://bucket/prefix'
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/${ARCH}-${LEVEL} \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -a architecture ${ARCH} \
       train
```
