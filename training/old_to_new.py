#!/usr/bin/env python3

# The MIT License (MIT)
# =====================
#
# Copyright © 2023
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

import argparse
import json
from pystac import Catalog
from pystac.stac_io import DefaultStacIO, StacIO
import shapely
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem
import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def pystac_workaround(uri):
    if uri.startswith('/vsizip/') and not uri.startswith('/vsizip//'):
        uri = uri.replace('/vsizip/', '/vsizip//')
    if uri.startswith('/vsitar/vsigzip/') and not uri.startswith('/vsitar/vsigzip//'):
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


if __name__ == "__main__":

    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Compute the AOI covered by a(n old-style GroundWork) STAC export for use with RasterVision ≥ 0.21")
    parser.add_argument("in_uri", type=str, help="Path to the STAC catalog")
    parser.add_argument("out_file", type=str, help="Where to deposite the computed AOI")
    args = parser.parse_args()
    # yapf: enable

    in_uri = args.in_uri.replace("s3://", "/vsizip/vsis3/")
    catalog = Catalog.from_file(root_of_tarball(in_uri))
    catalog.make_all_asset_hrefs_absolute()
    catalog = next(catalog.get_children())
    children = list(catalog.get_children())
    labels = next(filter(lambda child: "label" in str.lower(child.description), children))
    labels_item = next(labels.get_items())
    labels_href = labels_item.to_dict().get('assets').get('label').get('href')
    labels_data = VsiFileSystem.read_str(pystac_workaround(labels_href))
    labels_data = json.loads(labels_data)
    geoms = list(map(lambda s: shape(s.get('geometry')).buffer(0), labels_data.get('features')))
    aoi = union_polygon = unary_union(geoms)

    geojson_feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(aoi),
    }

    with open(args.out_file, 'w') as f:
        json.dump(geojson_feature, f)
