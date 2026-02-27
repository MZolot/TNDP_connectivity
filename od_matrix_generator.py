from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd

from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.preprocessing.imputing import impute_population

from services_config import SERVICE_CAPACITY_RANGE


def assign_services_capacity(services_gdf, capacity_range_dict=SERVICE_CAPACITY_RANGE):
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

    # print((polygons_split['area'] < 5).sum())
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
        for j in range(n):
            pop_avg = (P[i] + P[j]) / 2
            attr_avg = (A[i] + A[j]) / 2
            T[i, j] = pop_avg * attr_avg * decay[i, j]

    for i in range(n):
        row_sum = T[i, :].sum()
        if row_sum > 0:
            T[i, :] = T[i, :] * P[i] / row_sum

    od_matrix = pd.DataFrame(
        T,
        index=blocks_gdf['block_id'],
        columns=blocks_gdf['block_id']
    )

    od_matrix = (od_matrix + od_matrix.T) / 2

    od_matrix = od_matrix.loc[~(od_matrix == 0).all(axis=1), :]
    od_matrix = od_matrix.loc[:, ~(od_matrix == 0).all(axis=0)]

    od_matrix = od_matrix * 100
    return od_matrix


def generate_connector_od_matrix(graph, block_od_matrix):
    # список коннекторов
    connectors = [n for n, data in graph.nodes(data=True) if str(n).endswith("_connect")]
    
    # создаём пустую матрицу
    connector_od = pd.DataFrame(0, index=connectors, columns=connectors, dtype=float)
    
    # проходим по всем парам коннекторов
    for from_node in tqdm(connectors, desc="Building connector OD matrix"):
        from_blocks = graph.nodes[from_node].get('blocks', [])
        for to_node in connectors:
            to_blocks = graph.nodes[to_node].get('blocks', [])
            
            # суммируем спрос между блоками
            total_demand = sum(
                block_od_matrix.loc[f_block, t_block]
                for f_block in from_blocks
                for t_block in to_blocks
            )
            connector_od.loc[from_node, to_node] = total_demand
    
    return connector_od


def describe(od_matrix):
    od_matrix_values = od_matrix.values.flatten()
    od_matrix_series = pd.Series(od_matrix_values)

    return od_matrix_series.describe()