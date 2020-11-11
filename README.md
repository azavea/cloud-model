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
       -w /workdir \
       raster-vision:cloud-model bash
```

# Invoke #

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
