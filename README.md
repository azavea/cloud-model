![The Clouds](https://upload.wikimedia.org/wikipedia/commons/4/46/Socrates_in_a_basket.jpg)

# Build Image #

```bash
docker build -t raster-vision:cloud-model -f Dockerfile .
```

# Run Container #

```bash
docker run --name cloud-model -it --rm --runtime=nvidia --shm-size 16G \
       -v $HOME/.aws:/root/.aws:ro \
       -v $(pwd):/workdir \
       -v $HOME/local/src/raster-vision:/opt/src:ro \
       -w /workdir \
       raster-vision:cloud-model bash
```

# Invoke #

## Local ##

```bash
rastervision run inprocess /workdir/pipeline.py \
       -a root_uri /tmp/xxx \
       -a analyze_uri /tmp/xxx/analyze \
       -a chip_uri /tmp/xxx/chips \
       -a json catalogs.json \
       -a epochs 2 \
       -a batch_sz 2 \
       analyze chip
```

## Cloud ##

```bash
(LEVEL='L1C' ; \
ROOT="s3://bucket/prefix" ; \
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/0 \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -s 800 \
       chip)
```

```bash
(LEVEL='L1C' ; \
ROOT="s3://bucket/prefix" ; \
rastervision run batch /workdir/pipeline.py \
       -a root_uri ${ROOT}/0 \
       -a analyze_uri ${ROOT}/${LEVEL}/analyze \
       -a chip_uri ${ROOT}/${LEVEL}/chips \
       -a json catalogs.json \
       -a level ${LEVEL} \
       -s 800 \
       train)
```
