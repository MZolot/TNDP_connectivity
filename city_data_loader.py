import pandas as pd
import geopandas as gpd
import osmnx as ox

from iduedu import get_4326_boundary, get_intermodal_graph, get_drive_graph, get_walk_graph

from services_config import SERVICE_OSM_TAGS

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


def get_streets_graph(
        boundary_polygon,
        graph_type,
        keep_largest_subgraph=True,
        clip_by_territory=True):

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

    return


def get_services(boundary_polygon, tags_dict=SERVICE_OSM_TAGS):
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
