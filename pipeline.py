# flake8: noqa

import hashlib
from functools import partial

from pystac import STAC_IO, Catalog  #, Collection, Item, MediaType
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.data import (
    ClassConfig, DatasetConfig, GeoJSONVectorSourceConfig,
    RasterioSourceConfig, RasterizedSourceConfig, RasterizerConfig,
    SceneConfig, SemanticSegmentationLabelSourceConfig, CastTransformerConfig,
    StatsTransformerConfig)
from rastervision.core.rv_pipeline import *
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def noop_write_method(uri, txt):
    pass


def pystac_workaround(uri):
    if uri.startswith('/vsizip/') and not uri.startswith('/vsizip//'):
        uri = uri.replace('/vsizip/', '/vsizip//')
    if uri.startswith(
            '/vsitar/vsigzip/') and not uri.startswith('/vsitar/vsigzip//'):
        uri = uri.replace('/vsitar/vsigzip/', '/vsitar/vsigzip//')

    return uri
    return VsiFileSystem.read_str(uri)


STAC_IO.read_text_method = \
    lambda uri: VsiFileSystem.read_str(pystac_workaround(uri))
STAC_IO.write_text_method = noop_write_method


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

    label_dict = labels_item.to_dict()
    label_dict['properties']['class_id'] = None
    aoi_geometries = [label_dict]

    return (imagery_href, labels_href, aoi_geometries)


def hrefs_to_sceneconfig(
        imagery: str,
        labels: Optional[str],
        aoi: str,
        name: str,
        channel_order: Union[List[int], str],
        class_id_filter_dict: Dict[int, str],
        extent_crop: Optional[CropOffsets] = None) -> SceneConfig:

    transformers = [CastTransformerConfig(to_dtype='np.float32')]
    image_source = RasterioSourceConfig(
        uris=[imagery],
        allow_streaming=True,
        channel_order=channel_order,
        transformers=transformers,
        extent_crop=extent_crop,
    )

    label_vector_source = GeoJSONVectorSourceConfig(
        uri=labels,
        class_id_to_filter=class_id_filter_dict,
        default_class_id=1)
    label_raster_source = RasterizedSourceConfig(
        vector_source=label_vector_source,
        rasterizer_config=RasterizerConfig(background_class_id=0,
                                           all_touched=True))
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=label_raster_source)

    return SceneConfig(id=name,
                       aoi_geometries=aoi,
                       raster_source=image_source,
                       label_source=label_source)


def get_scenes(
    json_file: str,
    channel_order: Sequence[int],
    class_config: ClassConfig,
    class_id_filter_dict: dict,
    level: str,
    train_crops: List[CropOffsets] = [],
    val_crops: List[CropOffsets] = []
) -> Tuple[List[SceneConfig], List[SceneConfig]]:

    assert (level in ['L1C', 'L2A'])
    train_scenes = []
    val_scenes = []
    with open(json_file, 'r') as f:
        for catalog_imagery in json.load(f):
            catalog = catalog_imagery.get('catalog')
            catalog = catalog.strip()
            catalog = catalog.replace("s3://", "/vsizip/vsis3/")
            _, labels, aoi = hrefs_from_catalog(
                Catalog.from_file(root_of_tarball(catalog)))
            imagery = catalog_imagery.get('imagery')
            imagery = imagery.replace('L1C-0.tif', f"{level}-0.tif")
            h = hashlib.sha256(catalog.encode()).hexdigest()
            print('imagery', imagery)
            print('labels', labels)
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
               json,
               chip_sz=512,
               batch_sz=16,
               epochs=33,
               preshrink=1,
               level='L1C'):

    chip_sz = int(chip_sz)
    epochs = int(epochs)
    batch_sz = int(batch_sz)
    preshrink = int(preshrink)

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

    scenes = get_scenes(json,
                        channel_order,
                        class_config,
                        class_id_filter_dict,
                        level,
                        train_crops=train_crops,
                        val_crops=val_crops)

    train_scenes, validation_scenes = scenes

    print(f"{len(train_scenes)} training scenes")
    print(f"{len(validation_scenes)} validation scenes")

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=validation_scenes,
    )

    model = SemanticSegmentationModelConfig(
        external_def=ExternalModuleConfig(github_repo='jamesmcclain/CheapLab',
                                          name='cheaplab',
                                          entrypoint='make_cheaplab_model',
                                          entrypoint_kwargs={
                                              'preshrink': preshrink,
                                              'num_channels': num_channels
                                          }))

    external_loss_def = ExternalModuleConfig(
        github_repo='jamesmcclain/CheapLab',
        name='bce_loss',
        entrypoint='make_bce_loss',
        force_reload=False,
        entrypoint_kwargs={})

    backend = PyTorchSemanticSegmentationConfig(
        model=model,
        solver=SolverConfig(lr=1e-4,
                            num_epochs=epochs,
                            batch_sz=batch_sz,
                            external_loss_def=external_loss_def,
                            ignore_last_class='force'),
        log_tensorboard=False,
        run_tensorboard=False,
        num_workers=0,
        preview_batch_limit=8)

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding)

    return SemanticSegmentationConfig(root_uri=root_uri,
                                      analyze_uri=analyze_uri,
                                      chip_uri=chip_uri,
                                      dataset=dataset,
                                      backend=backend,
                                      train_chip_sz=chip_sz,
                                      predict_chip_sz=chip_sz,
                                      chip_options=chip_options,
                                      chip_nodata_threshold=.75,
                                      img_format='npy',
                                      label_format='npy')
