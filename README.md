![The Clouds](https://upload.wikimedia.org/wikipedia/commons/4/46/Socrates_in_a_basket.jpg)

# Build Image #

```bash
docker build . --tag raster-vision:pytorch-pystac-7abe1e8
```

# Run Container #

```bash
docker run --runtime=nvidia -it --rm -v $HOME/.aws:/root/.aws:ro -w /workdir raster-vision:pytorch-pystac-7abe1e8 bash
```

# Invoke Raster-Vision #

```bash
rastervision run batch /workdir/pipeline.py -a root_uri s3://mybucket/mypath/hand/ -a catalogs /vsitar//vsis3/mybucket/catalog.tar -a epochs 1
```
