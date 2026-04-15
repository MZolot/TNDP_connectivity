from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd

from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.preprocessing.imputing import impute_population

from .services_config import SERVICE_CAPACITY_RANGE


def assign_services_capacity(services_gdf, capacity_range_dict=SERVICE_CAPACITY_RANGE):
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


def assign_buildings_population(buildings, total_population, living_demand = 20):
    buildings_gdf = buildings.to_crs(32645)
    buildings_gdf['area'] = buildings_gdf.to_crs(32645).geometry.area
    buildings_gdf.sort_values(by=['is_living'])
    buildings_gdf['number_of_floors'] = buildings_gdf['number_of_floors'].fillna(
        1.0)

    build_population = impute_buildings(buildings.to_crs(
        32645), default_living_demand=living_demand).to_crs(32645)
    build_population['population'] = 0
    territory_population = impute_population(
        build_population, total_population)

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

    points_joined = points_joined.loc[:, ~points_joined.columns.duplicated()]
    polygons_split = polygons_split.loc[:,
                                        ~polygons_split.columns.duplicated()]

    services_processed = pd.concat(
        [points_joined, polygons_split],
        ignore_index=True
    )

    return services_processed


def generate_od_matrix(blocks_gdf, services_gdf):
    blocks_gdf = blocks_gdf.copy()

    blocks_gdf['population'] = blocks_gdf['population'].fillna(0)
    attraction_series = (
        services_gdf
        .groupby('block_id')['capacity_part']
        .sum()
    )
    blocks_gdf['attraction'] = blocks_gdf['block_id'].map(
        attraction_series).fillna(0)

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
    P_i = P[:, None]
    P_j = P[None, :]
    A_i = A[:, None]
    A_j = A[None, :]

    pop_avg = (P_i + P_j) / 2
    attr_avg = (A_i + A_j) / 2

    T = pop_avg * attr_avg * decay

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T * (P[:, None] / row_sums)

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


def generate_od_matrix_ipf(
    blocks_gdf,
    services_gdf,
    alpha=1.0,
    epsilon=1e-6,
    max_iter=100,
    tol=1e-6,
    background=0.05,
    threshold=0.1
):
    blocks = blocks_gdf.copy()
    P = blocks['population'].fillna(0).values
    capacity_per_block = services_gdf.groupby(
        'block_id')['capacity_part'].sum()
    blocks['capacity'] = blocks['block_id'].map(capacity_per_block).fillna(0)
    C = blocks['capacity'].values
    C = C + epsilon
    C = C * (P.sum() / C.sum())
    centroids = blocks.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in centroids])
    dist_matrix = cdist(coords, coords, metric='euclidean')
    K = np.exp(-alpha * dist_matrix) + background
    T = K.copy()
    for _ in range(max_iter):
        T_prev = T.copy()
        row_sums = T.sum(axis=1)
        T = (T.T * (P / (row_sums + epsilon))).T
        col_sums = T.sum(axis=0)
        T = T * (C / (col_sums + epsilon))
        if np.max(np.abs(T - T_prev)) < tol:
            break
    T = (T + T.T) / 2
    T[T < threshold] = 0
    od_df = pd.DataFrame(
        T, index=blocks['block_id'], columns=blocks['block_id'])
    return od_df


def generate_connector_od_matrix(graph, block_od_matrix):
    connectors = [n for n, data in graph.nodes(
        data=True) if str(n).endswith("_connect")]

    connector_od = pd.DataFrame(
        0, index=connectors, columns=connectors, dtype=float)

    for from_node in tqdm(connectors, desc="Building connector OD matrix"):
        from_blocks = graph.nodes[from_node].get('blocks', [])
        for to_node in connectors:
            to_blocks = graph.nodes[to_node].get('blocks', [])

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
