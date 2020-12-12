![The Clouds](https://upload.wikimedia.org/wikipedia/commons/4/46/Socrates_in_a_basket.jpg)

# Training Build Image #

```bash
docker build -t cloud-model -f Dockerfile .
```

# Run Container #

```bash
docker run --name cloud-model -it --rm --runtime=nvidia --shm-size 16G \
       -v $HOME/.aws:/root/.aws:ro \
       cloud-model bash
```

# Invoke #

## Local ##

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

## On AWS ##

### Chip ###

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

### Train ###

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
