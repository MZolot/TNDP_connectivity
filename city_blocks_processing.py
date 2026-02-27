from collections import defaultdict
from typing import cast
from shapely.geometry.base import BaseGeometry
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from shapely.ops import split
from tqdm.auto import tqdm

CRS = 3857


def get_centroids(blocks):
    blocks = blocks.to_crs(CRS)
    blocks["centroid_node"] = blocks.geometry.representative_point()
    return blocks.set_geometry("centroid_node")


def get_edge_intersection_points(blocks, pedestrian_edges):
    blocks_3857 = blocks.to_crs(epsg=CRS)
    ped_edges_3857 = pedestrian_edges.to_crs(epsg=CRS)

    connector_points = []

    for idx, block in tqdm(blocks_3857.iterrows(), total=len(blocks_3857), desc="   Getting intersections for blocks", leave=False):
        boundary = block.geometry.boundary

        for jdx, edge in ped_edges_3857.iterrows():
            line = edge.geometry
            if line.intersects(boundary):
                inter = line.intersection(boundary)

                if inter.geom_type == 'Point':
                    connector_points.append(
                        {'geometry': inter, 'block_id': idx})
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter.geoms:
                        connector_points.append(
                            {'geometry': pt, 'block_id': idx})

    connectors_gdf = gpd.GeoDataFrame(connector_points, crs=CRS)
    return connectors_gdf


def get_connector_points(blocks, pedestrian_nodes, buffer=10):
    bl = blocks.to_crs(CRS)
    ped_nodes = pedestrian_nodes.to_crs(CRS)

    bl["boundary"] = bl.geometry.boundary
    bl["boundary_buffer"] = bl["boundary"].buffer(buffer)
    buffers = bl.set_geometry("boundary_buffer")
    nodes_in_buffer = gpd.sjoin(
        ped_nodes,
        buffers,
        predicate="within",
        how="inner"
    )

    return nodes_in_buffer


def _filter_close_points(points_gdf, min_dist):
    selected = []

    for idx, row in points_gdf.iterrows():
        point = row.geometry

        if all(point.distance(sel.geometry) > min_dist for sel in selected):
            selected.append(row)

    if selected:
        return gpd.GeoDataFrame(selected, crs=points_gdf.crs)
    else:
        return gpd.GeoDataFrame(columns=points_gdf.columns, crs=points_gdf.crs)


def filter_close_nodes(nodes_in_blocks_buffer, nodes_distance=40, crs=CRS):
    filtered_nodes = []

    for block_id, group in nodes_in_blocks_buffer.groupby("block_id"):
        cleaned = _filter_close_points(group, min_dist=nodes_distance)
        cleaned["block_id"] = block_id
        filtered_nodes.append(cleaned)

    filtered_nodes = gpd.GeoDataFrame(
        pd.concat(filtered_nodes, ignore_index=True),
        crs=crs
    )

    return filtered_nodes


def merge_close_nodes(nodes_gdf, blocks_gdf, node_merge_dist=2, block_boundary_dist=10):
    nodes = nodes_gdf.to_crs(CRS)
    blocks = blocks_gdf.to_crs(CRS)

    coords = np.array([[p.x, p.y] for p in nodes.geometry])
    clustering = DBSCAN(eps=node_merge_dist, min_samples=1).fit(coords)
    nodes['cluster'] = clustering.labels_

    merged_nodes = nodes.groupby('cluster').geometry.apply(lambda g: Point(
        np.mean([p.x for p in g]), np.mean([p.y for p in g]))).reset_index() # pyright: ignore[reportArgumentType]

    blocks["block_id"] = blocks.index
    block_lists = []

    for pt in merged_nodes.geometry:
        close_blocks = blocks[blocks.geometry.boundary.distance(
            pt) <= block_boundary_dist]['block_id'].tolist()
        block_lists.append(close_blocks)

    merged_nodes['block_ids'] = block_lists
    return merged_nodes


def limit_connectors(nodes_gdf, blocks_with_centroids, max_k=5):
    centroid_dict = blocks_with_centroids["centroid_node"].to_dict()

    limited = []

    for block_id, group in nodes_gdf.groupby("block_id"):
        centroid = centroid_dict[block_id]

        group = group.copy()
        group["dist_to_centroid"] = group.geometry.distance(centroid)
        group = group.sort_values("dist_to_centroid")

        limited.append(group.head(max_k))

    return gpd.GeoDataFrame(
        pd.concat(limited, ignore_index=True),
        crs=nodes_gdf.crs
    )


def limit_connectors_segmented(nodes_gdf, blocks_with_centroids, max_k=5):
    selected_nodes = {}

    for block_id, block in blocks_with_centroids.iterrows():
        group = nodes_gdf[nodes_gdf['block_ids'].apply(
            lambda x: block_id in x)].copy()
        if group.empty:
            continue

        boundary = block.geometry.boundary
        segment_length = boundary.length / max_k
        segment_points = [boundary.interpolate(
            i * segment_length) for i in range(max_k)]

        for pt in segment_points:
            if group.empty:
                break
            distances = group.geometry.distance(pt)
            nearest_idx = distances.idxmin()
            nearest_node = group.loc[nearest_idx]

            geom_key = nearest_node.geometry.wkt

            if geom_key in selected_nodes:
                existing_blocks = set(selected_nodes[geom_key]['block_ids'])
                existing_blocks.update(nearest_node['block_ids'])
                selected_nodes[geom_key]['block_ids'] = list(existing_blocks)
            else:
                selected_nodes[geom_key] = nearest_node.to_dict()
                if not isinstance(selected_nodes[geom_key]['block_ids'], list):
                    selected_nodes[geom_key]['block_ids'] = list(
                        selected_nodes[geom_key]['block_ids'])
            group = group.drop(nearest_idx)

    if selected_nodes:
        return gpd.GeoDataFrame(list(selected_nodes.values()), crs=nodes_gdf.crs)
    else:
        return gpd.GeoDataFrame(columns=nodes_gdf.columns, crs=nodes_gdf.crs)


def assign_buildings_to_blocks(buildings_gdf, blocks_gdf):
    buildings_with_blocks = gpd.sjoin(
        buildings_gdf,
        blocks_gdf,
        how="left",
        predicate="intersects"
    )

    return buildings_with_blocks


def mean_graph_distance_to_connector_optimized(
    G,
    buildings,
    connector_point,
    block,
    weight="length_meter"
):
    if buildings.empty:
        return block['centroid_node'].distance(connector_point)

    connector_node = str(ox.distance.nearest_nodes(
        G,
        X=connector_point.x,
        Y=connector_point.y
    ))

    try:
        lengths = nx.single_source_dijkstra_path_length(
            G,
            connector_node,
            weight=weight
        )
    except TypeError:
        # print(f'Incorrect corrector node or smth: {connector_node}')
        return block['centroid_node'].distance(connector_point)

    building_points = buildings.geometry.centroid

    building_nodes = ox.distance.nearest_nodes(
        G,
        X=building_points.x.values,
        Y=building_points.y.values
    )

    building_points = buildings.geometry.centroid

    building_nodes = ox.distance.nearest_nodes(
        G,
        X=building_points.x.values,
        Y=building_points.y.values
    )

    distances = []

    for node in building_nodes:
        if node in lengths:
            distances.append(lengths[node])

    if len(distances) == 0:
        return block['centroid_node'].distance(connector_point)

    return float(np.mean(distances))


def build_block_graph(blocks_gdf, buildings_gdf, G_pedestrian, connectors_gdf, weight="length_meter"):
    blocks = blocks_gdf.to_crs(CRS)
    buildings = buildings_gdf.to_crs(CRS)

    G_quarters = nx.MultiDiGraph()
    buildings_with_blocks = assign_buildings_to_blocks(buildings, blocks)
    buildings_in_blocks = {}

    for idx, row in blocks.iterrows():
        block_id = row['block_id']
        centroid = row['centroid_node']
        x = centroid.x
        y = centroid.y
        G_quarters.add_node(f'{block_id}_block',
                            geometry=centroid, x=x, y=y, type='centroid')

        buildings_in_block = buildings_with_blocks[buildings_with_blocks["block_id"] == block_id]
        buildings_in_blocks[block_id] = buildings_in_block

    for idx, row in tqdm(blocks.iterrows(), total=len(blocks), desc="   Building graph for blocks", leave=False):
        block_id = row['block_id']
        centroid = row['centroid_node']

        buildings = buildings_in_blocks.get(block_id)

        connectors = connectors_gdf[connectors_gdf['block_ids'].apply(
            lambda x: block_id in x)]

        if connectors.empty:
            G_quarters.remove_node(f'{block_id}_block')
            continue

        for i, connector in connectors.iterrows():
            connector_geom = connector.geometry
            mean_dist = mean_graph_distance_to_connector_optimized(
                G=G_pedestrian,
                buildings=buildings,
                connector_point=connector_geom,
                block=row,
                weight=weight
            )
            block_node_name = f'{block_id}_block'
            connector_id = f"{connector.name}_connect"
            x = connector_geom.x
            y = connector_geom.y
            line = LineString([centroid, connector_geom])

            if connector_id not in G_quarters:
                G_quarters.add_node(connector_id, geometry=connector_geom,
                                    x=x, y=y, type='connector', blocks=[block_node_name])
            else:
                G_quarters.nodes[connector_id]['blocks'].append(
                    block_node_name)
            G_quarters.add_edge(block_node_name, connector_id,
                                length_meter=mean_dist, geometry=line)

    G_quarters.graph['crs'] = 'EPSG:3857'

    return G_quarters


def merge_blocks_into_streets(G_streets, G_blocks, epsilon=5):
    # --- 0. Приводим граф улиц к MultiGraph (неориентированный) ---
    G_streets_undir = nx.MultiGraph(G_streets)

    # --- 1. Преобразуем граф улиц в GeoDataFrame и CRS 3857 ---
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_streets_undir)
    nodes_gdf_3857 = nodes_gdf.to_crs(epsg=3857)
    edges_gdf_3857 = edges_gdf.to_crs(epsg=3857).reset_index()

    # --- 2. Создаём новый граф ---
    G_new = nx.MultiDiGraph()

    # копируем узлы улиц
    for node_id, data in nodes_gdf_3857.iterrows():
        attrs = {str(k): v for k, v in data.drop(['x', 'y']).to_dict().items()}
        G_new.add_node(node_id, x=data['x'], y=data['y'], **attrs)

    # копируем рёбра улиц
    for idx, row in edges_gdf_3857.iterrows():
        u, v, key = row['u'], row['v'], row['key']
        data = row.drop(['u', 'v', 'key']).to_dict()
        attrs = {str(k): v for k, v in data.items()}
        G_new.add_edge(u, v, **attrs)

    # --- 3. Группируем коннекторы по ближайшему ребру ---
    connectors_by_edge = defaultdict(list)
    outside_connectors = []

    for node_id, data in G_blocks.nodes(data=True):
        if not str(node_id).endswith("_connect"):
            continue

        point = Point(data["x"], data["y"])
        edges_nonnull = edges_gdf_3857[edges_gdf_3857.geometry.notnull()]
        distances = edges_nonnull.distance(point)
        if not distances.empty:
            nearest_edge_idx = distances.idxmin()
            min_dist = distances.loc[nearest_edge_idx]
        else:
            min_dist = float('inf')

        if min_dist <= epsilon:
            u = edges_nonnull.loc[nearest_edge_idx, "u"]
            v = edges_nonnull.loc[nearest_edge_idx, "v"]
            connectors_by_edge[(u, v)].append(node_id)
        else:
            outside_connectors.append(node_id)
            # добавляем узел вне улицы
            G_new.add_node(
                node_id,
                x=data["x"],
                y=data["y"],
                blocks=data.get("blocks", [])
            )

    # --- 4. Вставляем коннекторы на ребра ---
    for (u, v), conns in connectors_by_edge.items():
        street_edge = edges_gdf_3857[
            (edges_gdf_3857['u'] == u) & (edges_gdf_3857['v'] == v)
        ].iloc[0]
        geom = cast(BaseGeometry, street_edge.geometry)

        # сортируем коннекторы по проекции на линию
        conns_sorted = sorted(conns, key=lambda nid: geom.project(
            Point(G_blocks.nodes[nid]["x"], G_blocks.nodes[nid]["y"])))

        # строим цепочку узлов: u → c1 → c2 → ... → v
        nodes_chain = [u] + conns_sorted + [v]
        points_chain = [Point(geom.coords[0])] + \
                       [Point(G_blocks.nodes[nid]["x"], G_blocks.nodes[nid]["y"]) for nid in conns_sorted] + \
                       [Point(geom.coords[-1])]

        # удаляем старое ребро
        keys_to_remove = list(G_new[u][v].keys())
        for k in keys_to_remove:
            G_new.remove_edge(u, v, k)

        # добавляем новые ребра между цепочкой узлов
        for i in range(len(nodes_chain)-1):
            n1, n2 = nodes_chain[i], nodes_chain[i+1]
            p1, p2 = points_chain[i], points_chain[i+1]

            # добавляем узлы, если ещё нет, и для коннекторов сохраняем blocks
            for n, p in [(n1, p1), (n2, p2)]:
                if n not in G_new:
                    attrs = {}
                    if str(n).endswith("_connect"):
                        attrs["blocks"] = G_blocks.nodes[n].get("blocks", [])
                    G_new.add_node(n, x=p.x, y=p.y, **attrs)

            # добавляем ребро
            line_seg = LineString([p1, p2])
            G_new.add_edge(n1, n2, geometry=line_seg, length=line_seg.length)

    # --- 5. Добавляем блоки, соединённые с коннекторами ---
    for node_id, data in G_blocks.nodes(data=True):
        if not str(node_id).endswith("_connect"):
            continue

        neighbors = [n for n in G_blocks.predecessors(
            node_id) if str(n).endswith("_block")]
        for block_node in neighbors:
            x_block, y_block = G_blocks.nodes[block_node]["x"], G_blocks.nodes[block_node]["y"]
            if block_node not in G_new:
                G_new.add_node(block_node, x=x_block, y=y_block)

            # добавляем ребра block → connect
            edge_data_dict = G_blocks.get_edge_data(block_node, node_id)
            if isinstance(edge_data_dict, dict) and any(isinstance(v, dict) for v in edge_data_dict.values()):
                edge_items = edge_data_dict.items()  # MultiDiGraph
            else:
                edge_items = [(None, edge_data_dict)]  # обычный граф

            for edge_key, edge_data in edge_items:
                connector_geom = edge_data["geometry"]
                G_new.add_edge(
                    block_node,
                    node_id,
                    geometry=connector_geom,
                    length_meter=connector_geom.length,
                    weight=connector_geom.length,
                    type="connector"
                )

    G_new.graph['crs'] = 'EPSG:3857'
    return G_new


def get_blocks_graph(graph: nx.MultiDiGraph, blocks, streets_graph, buildings, node_merge_dist=70, connectors_count=5):
    ped_nodes, ped_edges = ox.graph_to_gdfs(graph)
    ped_nodes = ped_nodes.reset_index(drop=True)

    steps = [
        ("Getting existing connectors",
         lambda: get_connector_points(blocks, ped_nodes)),
        ("Getting edge intersections",
         lambda: get_edge_intersection_points(blocks, ped_edges)),
        ("Filtering nodes", None),
        ("Merging nodes", None),
        ("Filtering connectors", None),
        ("Building blocks graph", None),
        ("Merging with streets graph", None),
    ]

    with tqdm(total=len(steps), desc="Building blocks graph") as pbar:

        pbar.set_description("Getting existing connectors")
        nodes_in_buffer = get_connector_points(blocks, ped_nodes)
        pbar.update(1)

        pbar.set_description("Getting edge intersections")
        intersection_nodes = get_edge_intersection_points(blocks, ped_edges)
        pbar.update(1)

        buffer_nodes = nodes_in_buffer.rename(
            columns={'index_right': 'block_id'})
        intersection_nodes = intersection_nodes.rename(
            columns={'block_idx': 'block_id'})
        buffer_nodes = buffer_nodes.loc[:, ~buffer_nodes.columns.duplicated()]
        intersection_nodes = intersection_nodes.loc[:,
                                                    ~intersection_nodes.columns.duplicated()]

        all_nodes = gpd.GeoDataFrame(
            pd.concat([buffer_nodes, intersection_nodes]),
            crs=buffer_nodes.crs
        )

        pbar.set_description("Filtering nodes")
        filtered_nodes = filter_close_nodes(all_nodes)
        pbar.update(1)

        pbar.set_description("Merging nodes")
        merged_nodes = merge_close_nodes(
            filtered_nodes, blocks, node_merge_dist=node_merge_dist
        )
        pbar.update(1)

        pbar.set_description("Filtering connectors")
        final_connectors = limit_connectors_segmented(
            merged_nodes, blocks, max_k=connectors_count
        )
        pbar.update(1)

        pbar.set_description("Building graph")
        blocks = blocks.to_crs(CRS)
        blocks["block_id"] = blocks.index
        blocks["centroid_node"] = blocks.geometry.representative_point()

        G_blocks = build_block_graph(
            blocks, buildings, graph, final_connectors, weight='length_meter'
        )
        pbar.update(1)

        G_fin = merge_blocks_into_streets(streets_graph, G_blocks)
        pbar.update(1)

    return G_fin
