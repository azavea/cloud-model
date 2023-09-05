# flake8: noqa

# The MIT License (MIT)
# =====================
#
# Copyright © 2020-2023
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import hashlib
import json
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import rasterio as rio
from pystac import Catalog
from pystac.stac_io import DefaultStacIO, StacIO
from rastervision.core.backend import *
from rastervision.core.box import Box
from rastervision.core.data import (CastTransformerConfig, ClassConfig,
                                    ClassInferenceTransformerConfig,
                                    DatasetConfig, GeoJSONVectorSourceConfig,
                                    RasterioSourceConfig,
                                    RasterizedSourceConfig, RasterizerConfig,
                                    SceneConfig,
                                    SemanticSegmentationLabelSourceConfig)
from rastervision.core.rv_pipeline import *
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *

# rastervision run inprocess ./pipeline.py -a root_uri ${ROOT} -a analyze_uri ${ROOT}/analyze -a chip_uri ${ROOT}/chips -a json_catalog_list catalogs.json -a epochs 2 -a batch_sz 2 -a small_test True chip


def pystac_workaround(uri):
    if uri.startswith('/vsizip/') and not uri.startswith('/vsizip//'):
        uri = uri.replace('/vsizip/', '/vsizip//')
    if uri.startswith(
            '/vsitar/vsigzip/') and not uri.startswith('/vsitar/vsigzip//'):
        uri = uri.replace('/vsitar/vsigzip/', '/vsitar/vsigzip//')

    return uri


class CustomStacIO(DefaultStacIO):

    def read_text(self, source, *args, **kwargs) -> str:
        return VsiFileSystem.read_str(pystac_workaround(source))

    def write_text(self, dest, txt, *args, **kwargs) -> None:
        pass


StacIO.set_default(CustomStacIO)


def root_of_tarball(tarball: str) -> str:
    catalog_root = tarball
    while not (catalog_root.endswith('catalog.json')
               and catalog_root is not None):
        paths = VsiFileSystem.list_paths(catalog_root)
        if len(paths) > 1:
            paths = list(filter(lambda s: s.endswith('catalog.json'), paths))
        if len(paths) != 1:
            raise Exception("Unrecognizable Tarball")
        catalog_root = f"{paths[0]}"
    return catalog_root


def hrefs_from_catalog(catalog: Catalog) -> Tuple[str, str]:

    catalog.make_all_asset_hrefs_absolute()

    catalog = next(catalog.get_children())
    children = list(catalog.get_children())

    imagery = next(filter(lambda child: \
                          "image" in str.lower(child.description), children))
    imagery_item = next(imagery.get_items())
    imagery_assets = list(imagery_item.assets.values())
    imagery_href = pystac_workaround(imagery_assets[1].href)

    labels = next(filter(lambda child: \
                         "label" in str.lower(child.description), children))
    labels_item = next(labels.get_items())
    labels_href = pystac_workaround(
        next(iter(labels_item.assets.values())).href)

    return (imagery_href, labels_href)


def hrefs_to_sceneconfig(imagery: str,
                         labels: Optional[str],
                         aoi: str,
                         name: str,
                         channel_order: Union[List[int], str],
                         class_id_filter_dict: Dict[int, str],
                         extent_crop: Optional[Tuple] = None) -> SceneConfig:

    image_transformers = [CastTransformerConfig(to_dtype='float16')]
    with rio.open(pystac_workaround(imagery), 'r') as ds:
        width = ds.width
        height = ds.height
    ymin = int(extent_crop[1] * width)
    xmin = int(extent_crop[0] * height)
    ymax = int(extent_crop[3] * width)
    xmax = int(extent_crop[2] * height)
    bbox = Box(ymin, xmin, ymax, xmax)
    image_source = RasterioSourceConfig(
        uris=[imagery],
        allow_streaming=True,
        channel_order=channel_order,
        transformers=image_transformers,
        bbox=bbox,
    )

    label_transformers = [
        ClassInferenceTransformerConfig(
            class_id_to_filter=class_id_filter_dict, default_class_id=1)
    ]
    label_vector_source = GeoJSONVectorSourceConfig(
        uris=[labels],
        transformers=label_transformers,
    )

    label_raster_source = RasterizedSourceConfig(
        vector_source=label_vector_source,
        rasterizer_config=RasterizerConfig(background_class_id=0,
                                           all_touched=True))
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=label_raster_source)

    return SceneConfig(id=name,
                       aoi_uris=[aoi],
                       raster_source=image_source,
                       label_source=label_source)


def get_scenes(json_catalog_list: str,
               channel_order: Sequence[int],
               class_config: ClassConfig,
               class_id_filter_dict: dict,
               level: str,
               train_crops=[],
               val_crops=[]) -> Tuple[List[SceneConfig], List[SceneConfig]]:

    assert (level in ['L1C', 'L2A'])
    train_scenes = []
    val_scenes = []
    with open(json_catalog_list, 'r') as f:
        for catalog_list_item in json.load(f):  # XXX
            catalog = catalog_list_item.get('catalog')
            catalog = catalog.strip()
            catalog = catalog.replace("s3://", "/vsizip/vsis3/")
            _, labels = hrefs_from_catalog(
                Catalog.from_file(root_of_tarball(catalog)))
            imagery = catalog_list_item.get('imagery')
            imagery = imagery.replace('L1C-0.tif', f"{level}-0.tif")
            aoi = catalog_list_item.get('aoi')
            h = hashlib.sha256(catalog.encode()).hexdigest()
            print('imagery', imagery)
            print('labels', labels)
            print('aoi', aoi)
            make_scene = partial(hrefs_to_sceneconfig,
                                 imagery=imagery,
                                 labels=labels,
                                 aoi=aoi,
                                 channel_order=channel_order,
                                 class_id_filter_dict=class_id_filter_dict)
            for i, crop in enumerate(train_crops):
                scene = make_scene(name=f'{h}-train-{i}', extent_crop=crop)
                train_scenes.append(scene)
            for i, crop in enumerate(val_crops):
                scene = make_scene(name=f'{h}-val-{i}', extent_crop=crop)
                val_scenes.append(scene)
    return train_scenes, val_scenes


def get_config(runner,
               root_uri,
               analyze_uri,
               chip_uri,
               json_catalog_list,
               chip_sz=512,
               batch_sz=32,
               epochs=33,
               preshrink=1,
               small_test=False,
               architecture='cheaplab',
               level='L1C'):

    import sys, pdb
    def excepthook(type, value, tb):
        pdb.post_mortem(tb)
    sys.excepthook = excepthook

    chip_sz = int(chip_sz)
    epochs = int(epochs)
    batch_sz = int(batch_sz)
    preshrink = int(preshrink)
    assert (architecture in ['cheaplab', 'fpn-resnet18'])
    assert (level in ['L1C', 'L2A'])

    if level == 'L1C':
        channel_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif level == 'L2A':
        channel_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    num_channels = len(channel_order)

    class_config = ClassConfig(names=["background", "cloud"],
                               colors=["brown", "white"])

    class_id_filter_dict = {
        0: ['==', 'default', 'Background'],
        1: ['==', 'default', 'Cloud'],
    }

    train_crops = []
    val_crops = []
    for x in range(0, 5):
        for y in range(0, 5):
            x_start = x / 5.0
            x_end = 0.80 - x_start
            y_start = y / 5.0
            y_end = 0.80 - y_start
            crop = [x_start, y_start, x_end, y_end]
            if x == y:
                val_crops.append(crop)
            else:
                train_crops.append(crop)

    scenes = get_scenes(json_catalog_list,
                        channel_order,
                        class_config,
                        class_id_filter_dict,
                        level,
                        train_crops=train_crops,
                        val_crops=val_crops)

    train_scenes, validation_scenes = scenes

    if small_test:
        train_scenes = train_scenes[0:2]
        validation_scenes = validation_scenes[0:2]

    print(f"{len(train_scenes)} training scenes")
    print(f"{len(validation_scenes)} validation scenes")

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=validation_scenes,
    )

    if architecture == 'fpn-resnet18':
        external_def = ExternalModuleConfig(
            github_repo='AdeelH/pytorch-fpn:0.1',
            name='pytorch-fpn',
            entrypoint='make_segm_fpn_resnet',
            entrypoint_kwargs={
                'name': 'resnet18',
                'fpn_type': 'fpn',
                'num_classes': 2,
                'fpn_channels': 256,
                'in_channels': len(channel_order),
                'out_size': (chip_sz, chip_sz)
            })
    else:
        external_def = ExternalModuleConfig(
            github_repo='jamesmcclain/CheapLab:08d260b',
            name='cheaplab',
            entrypoint='make_cheaplab_model',
            entrypoint_kwargs={
                'preshrink': preshrink,
                'num_channels': num_channels
            })

    model = SemanticSegmentationModelConfig(external_def=external_def)

    external_loss_def = ExternalModuleConfig(
        github_repo='jamesmcclain/CheapLab:08d260b',
        name='bce_loss',
        entrypoint='make_bce_loss',
        force_reload=False,
        entrypoint_kwargs={})

    data = SemanticSegmentationImageDataConfig(img_sz=chip_sz,
                                               num_workers=0,
                                               preview_batch_limit=8)
    backend = PyTorchSemanticSegmentationConfig(
        model=model,
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=epochs,
            batch_sz=batch_sz,
            external_loss_def=external_loss_def,
            ignore_class_index=2,
        ),
        log_tensorboard=False,
        run_tensorboard=False,
        data=data)

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        analyze_uri=analyze_uri,
        chip_uri=chip_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        chip_nodata_threshold=.75,
    )
