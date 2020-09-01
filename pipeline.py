# flake8: noqa

import hashlib
import os
from os.path import join
from typing import Generator
import random

from pystac import STAC_IO, Catalog, Collection, Item, MediaType
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.data import (
    ClassConfig, DatasetConfig, GeoJSONVectorSourceConfig,
    RasterioSourceConfig, RasterizedSourceConfig, RasterizerConfig,
    SceneConfig, SemanticSegmentationLabelSourceConfig, StatsTransformerConfig)
from rastervision.core.rv_pipeline import *
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def noop_write_method(uri, txt):
    pass


def pystac_workaround(uri):
    if uri.startswith('/vsitar/') and not uri.startswith('/vsitar//'):
        uri = uri.replace('/vsitar/', '/vsitar//')
    return uri
    return VsiFileSystem.read_str(uri)


STAC_IO.read_text_method = \
    lambda uri: VsiFileSystem.read_str(pystac_workaround(uri))
STAC_IO.write_text_method = noop_write_method


def hrefs_to_sceneconfig(hrefs: Tuple[str, str], name: str,
                         channel_order: List[int],
                         class_id_filter_dict: Dict[int, str]) -> SceneConfig:
    raster_source = RasterioSourceConfig(
        uris=[hrefs[0]],
        channel_order=channel_order,
        transformers=[StatsTransformerConfig()])
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=hrefs[1],
                # class_id_to_filter=class_id_filter_dict, # XXX prevents rasterization
                default_class_id=0),
            rasterizer_config=RasterizerConfig(background_class_id=0)))
    return SceneConfig(
        id=name, raster_source=raster_source, label_source=label_source)


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


def hrefs_from_catalog(catalog: Catalog, channel_order: [int],
                       class_config: ClassConfig) -> Tuple[str, str]:

    catalog.make_all_asset_hrefs_absolute()

    catalog = next(catalog.get_children())
    children = list(catalog.get_children())

    imagery = next(
        filter(lambda child: "image" in str.lower(child.description),
               children))
    labels = next(
        filter(lambda child: "label" in str.lower(child.description),
               children))

    imagery_item = next(imagery.get_items())
    labels_item = next(labels.get_items())

    imagery_href = pystac_workaround(
        next(iter(imagery_item.assets.values())).href)
    labels_href = pystac_workaround(
        next(iter(labels_item.assets.values())).href)

    return (imagery_href, labels_href)


def get_config(runner, root_uri, catalogs, epochs):

    channel_ordering: [int] = [0, 1, 2]  # XXX should be multiband
    class_id_filter_dict = {
        0: ['==', 'default', 'Background'],
        1: ['==', 'default', 'Cloud'],
    }
    class_config: ClassConfig = ClassConfig(
        names=["Cloud", "Background"], colors=["cyan", "brown"])

    # Read STAC catalog(s)
    scenes = []
    if catalogs.endswith('.tar'):
        catalog = catalogs
        if not catalog.startswith('/vsitar/'):
            catalog = f"/vsitar/{catalog}"
        catalog_root = root_of_tarball(catalog)
        hrefs = hrefs_from_catalog(
            Catalog.from_file(catalog_root), channel_ordering, class_config)
        h = hashlib.sha256(catalog.encode()).hexdigest()
        scene = hrefs_to_sceneconfig(hrefs, h, channel_ordering,
                                     class_id_filter_dict)
        scenes.append(scene)
    else:
        for catalog in VsiFileSystem.list_paths(catalogs):
            if catalog.endswith('.tar'):
                print(f"Found {catalog}")
                catalog = f"/vsitar/{catalog}"
                catalog_root = root_of_tarball(catalog)
                hrefs = hrefs_from_catalog(
                    Catalog.from_file(catalog_root), channel_ordering,
                    class_config)
                h = hashlib.sha256(catalog.encode()).hexdigest()
                scene = hrefs_to_sceneconfig(hrefs, h, channel_ordering,
                                             class_id_filter_dict)
                scenes.append(scene)

    chip_sz = 512
    random.shuffle(scenes)
    n = 1  # XXX
    last_scene = max(1, len(scenes) - n)
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=scenes[0:last_scene],
        validation_scenes=[scenes[-1 * n]],
        # test_scenes=validation_scenes,
    )
    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(lr=1e-4, num_epochs=epochs, batch_sz=8),
    )
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
    )
