import numpy as np
import pandas as pd
import geopandas as gpd

from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.preprocessing.imputing import impute_population


def assign_services_capacity(services_gdf, capacity_range_dict):
    """
    Добавляет столбец 'capacity' к GeoDataFrame сервисов.
    Можно делать разные диапазоны для разных типов сервисов.
    """
    capacities = []
    for stype in services_gdf['service_type']:
        if stype in capacity_range_dict.keys():
            cap = np.random.randint(
                capacity_range_dict[stype][0], capacity_range_dict[stype][1])
        else:
            cap = np.random.randint(5, 100)  
        capacities.append(cap)

    services_gdf['capacity'] = capacities
    return services_gdf


def assign_buildings_population(buildings):
    buildings_gdf = buildings.to_crs(32645)
    buildings_gdf['area'] = buildings_gdf.to_crs(32645).geometry.area
    buildings_gdf.sort_values(by=['is_living'])
    buildings_gdf['number_of_floors'] = buildings_gdf['number_of_floors'].fillna(
        1.0)

    build_population = impute_buildings(buildings.to_crs(
        32645), default_living_demand=40).to_crs(32645)
    build_population['population'] = 0
    territory_population = impute_population(build_population, 80000)

    build_population['population'] = territory_population['population']
    build_population.sort_values(by=['is_living'])

    return build_population


def assign_blocks_population(blocks, buildings_with_population):
    blocks_gdf = blocks.to_crs(32645)
    blocks_gdf = blocks_gdf.reset_index(drop=False)
    blocks_gdf = blocks_gdf.rename(columns={'index': 'block_id'})

    joined = gpd.sjoin(
        buildings_with_population[['population', 'geometry']],
        blocks_gdf[['geometry']],
        how='left',
        predicate='within'
    )

    block_population = (
        joined
        .groupby('index_right')['population']
        .sum()
    )

    blocks_gdf['population'] = block_population
    blocks_gdf['population'] = blocks_gdf['population'].fillna(0)

    return blocks_gdf


def assign_blocks_to_services(services_gdf, blocks_gdf):
    services_gdf = services_gdf.reset_index(drop=True)
    services_gdf['service_id'] = services_gdf.index

    points = services_gdf[services_gdf.geometry.type.isin(['Point'])].copy()
    polygons = services_gdf[services_gdf.geometry.type.isin(
        ['Polygon', 'MultiPolygon'])].copy()

    points_joined = gpd.sjoin(
        points,
        blocks_gdf[['geometry']],
        how='left',
        predicate='within'
    )
    points_joined['capacity_part'] = points_joined['capacity']
    points_joined = points_joined[
        points_joined['index_right'].notna()
    ].copy()
    points_joined = points_joined.rename(
        columns={'index_right': 'block_id'}
    )

    polygons_split = gpd.overlay(
        polygons,
        blocks_gdf[['block_id', 'geometry']],
        how='intersection'
    )

    polygons_split['area'] = polygons_split.geometry.area

    total_area = (
        polygons_split
        .groupby('service_id')['area']
        .transform('sum')
    )

    polygons_split['capacity_part'] = (
        polygons_split['capacity'] *
        polygons_split['area'] /
        total_area
    )

    print((polygons_split['area'] < 5).sum())
    polygons_split = polygons_split[
        polygons_split['capacity_part'] >= 5
    ].copy()

    services_processed = pd.concat(
        [points_joined, polygons_split],
        ignore_index=True
    )

    return services_processed


def generate_od_matrix(blocks_gdf, services_gdf):
    blocks_gdf['population'] = blocks_gdf['population'].fillna(0)
    blocks_gdf['attraction'] = blocks_gdf['block_id'].map(
        services_gdf.groupby('block_id')['capacity_part'].sum()
    ).fillna(0)

    blocks_gdf['centroid'] = blocks_gdf.geometry.centroid

    n = len(blocks_gdf)
    dist_matrix = np.zeros((n, n))
    for i, ci in enumerate(blocks_gdf['centroid']):
        for j, cj in enumerate(blocks_gdf['centroid']):
            dist_matrix[i, j] = ci.distance(cj)

    beta = 0.001
    decay = np.exp(-beta * dist_matrix)

    A = blocks_gdf['attraction'].values
    P = blocks_gdf['population'].values

    T = np.zeros((n, n))
    for i in range(n):
        denom = np.sum(A * decay[i, :])
        if denom > 0:
            T[i, :] = P[i] * (A * decay[i, :] / denom)

    od_matrix = pd.DataFrame(
        T,
        index=blocks_gdf['block_id'],
        columns=blocks_gdf['block_id']
    )

    return od_matrix
