from blocksnet.blocks.postprocessing import postprocess_urban_blocks
from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx

from iduedu import get_4326_boundary, get_intermodal_graph, get_drive_graph, get_walk_graph

from .services_config import SERVICE_OSM_TAGS

CRS = 4326

IS_LIVING_TAGS = ['residential', 'house', 'apartments',
                  'detached', 'terrace', 'dormitory',
                  'semidetached_house']


def get_bounary_from_file(filename, crs=CRS):
    data = gpd.read_file(filename)
    # .convex_hull.buffer(0.001)
    polygon = data.to_crs(crs).union_all()
    return polygon


def get_boundary_from_osm(osm_id):
    polygon = get_4326_boundary(osm_id=osm_id)
    return polygon


def filter_edges(G, edge_tags):
    print(f'Filtering edges by tags: {edge_tags}')
    edges_to_keep = []

    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get('highway')

        if highway is None:
            continue

        if isinstance(highway, list):
            if any(h in edge_tags for h in highway):
                edges_to_keep.append((u, v, k))
        else:
            if highway in edge_tags:
                edges_to_keep.append((u, v, k))

    G_filtered = G.edge_subgraph(edges_to_keep).copy()
    
    isolated = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated)
    return G_filtered


def get_streets_graph(
        boundary_polygon,
        graph_type,
        keep_largest_subgraph=True,
        clip_by_territory=True,
        osm_tags=None):

    if graph_type == 'drive':
        graph = get_drive_graph(territory=boundary_polygon,
                                keep_largest_subgraph=keep_largest_subgraph,
                                clip_by_territory=clip_by_territory)
    elif graph_type == 'walk':
        graph = get_walk_graph(territory=boundary_polygon,
                               keep_largest_subgraph=keep_largest_subgraph,
                               clip_by_territory=clip_by_territory)
    else:
        graph = get_intermodal_graph(territory=boundary_polygon,
                                     keep_largest_subgraph=keep_largest_subgraph,
                                     clip_by_territory=clip_by_territory)

    if osm_tags is not None:
        return filter_edges(graph, osm_tags)
    else:
        return graph


def get_buildings(boundary_polygon):
    buildings = ox.features_from_polygon(
        boundary_polygon, tags={'building': True})
    buildings = buildings.reset_index(drop=True).to_crs(CRS)

    buildings['is_living'] = buildings['building'].apply(
        lambda b: b in IS_LIVING_TAGS)
    buildings['number_of_floors'] = pd.to_numeric(
        buildings['building:levels'], errors='coerce')

    buildings = buildings[buildings.geom_type.isin(
        ['Polygon', 'MultiPolygon'])]
    buildings = buildings.to_crs(CRS)

    return buildings


def get_services_from_osm(boundary_polygon, tags_dict=SERVICE_OSM_TAGS):
    all_services = []

    for service_name, tags_list in tags_dict.items():
        tags = {}
        for key, value in tags_list:
            if key in tags:
                if isinstance(tags[key], list):
                    tags[key].append(value)
                else:
                    tags[key] = [tags[key], value]
            else:
                tags[key] = value
        try:
            print(f"Загрузка: {service_name}")
            gdf = ox.features_from_polygon(boundary_polygon, tags)
        except Exception as e:
            print(f"Ошибка при загрузке {service_name}: {e}")
            continue

        if not gdf.empty:
            gdf = gdf[['geometry']].copy()
            gdf['service_type'] = service_name
            all_services.append(gdf)

    if all_services:
        services_gdf = gpd.GeoDataFrame(
            pd.concat(all_services, ignore_index=True), crs="EPSG:4326")
    else:
        services_gdf = gpd.GeoDataFrame(
            columns=['geometry', 'service_type'], crs="EPSG:4326")

    return services_gdf


def get_services_from_file(filename, crs=CRS):
    data = gpd.read_file(filename)
    return data


BC_TAGS = {
    'roads': {
        "highway": ["construction", "crossing", "living_street", "motorway", "motorway_link", "motorway_junction", "pedestrian", "primary", "primary_link", "raceway", "residential", "road", "secondary", "secondary_link", "services", "tertiary", "tertiary_link", "track", "trunk", "trunk_link", "turning_circle", "turning_loop", "unclassified",],
        "service": ["living_street", "emergency_access"]
    },
    'railways': {
        "railway": "rail"
    },
    'water': {
        'riverbank': True,
        'reservoir': True,
        'basin': True,
        'dock': True,
        'canal': True,
        'pond': True,
        'natural': ['water', 'bay'],
        'waterway': ['river', 'canal', 'ditch'],
        'landuse': 'basin',
        'water': 'lake'
    }
}


def get_blocks_by_blocksnet(boundary_polygon, boundary_gdf, G_roads, use_railways=False, use_waterways=False):
    _, roads_gdf = ox.graph_to_gdfs(G_roads)
    roads_gdf = roads_gdf.reset_index()
    roads_gdf.head(3)

    roads_osm = ox.features_from_polygon(boundary_polygon, BC_TAGS['roads'])
    roads_osm = roads_osm.reset_index()
    roads_osm = roads_osm[roads_osm.geom_type.isin(
        ['LineString', 'MultiLineString'])]

    if use_railways:
        railways = ox.features_from_polygon(
            boundary_polygon, BC_TAGS['railways'])
        railways = railways.reset_index()
    else:
        railways = None

    if use_waterways:
        waterways = ox.features_from_polygon(
            boundary_polygon, BC_TAGS['water'])
        waterways = waterways.reset_index()
    else:
        waterways = None

    lines_gdf, polygons_gdf = preprocess_urban_objects(
        roads_osm, railways, waterways)
    blocks = cut_urban_blocks(boundary_gdf, lines_gdf, polygons_gdf)

    blocks = postprocess_urban_blocks(blocks)

    return blocks
