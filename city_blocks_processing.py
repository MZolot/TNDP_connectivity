import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString

CRS = 3857


def get_centroids(blocks):
    blocks = blocks.to_crs(CRS)
    blocks["centroid_node"] = blocks.geometry.representative_point()
    return blocks.set_geometry("centroid_node")


def get_edge_intersection_points(blocks, pedestrian_edges):
    blocks_3857 = blocks.to_crs(epsg=CRS)
    ped_edges_3857 = pedestrian_edges.to_crs(epsg=CRS)

    connector_points = []

    for idx, block in blocks_3857.iterrows():
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
        group = nodes_gdf[nodes_gdf['block_ids'].apply(lambda x: block_id in x)].copy()
        if group.empty:
            continue

        boundary = block.geometry.boundary
        segment_length = boundary.length / max_k
        segment_points = [boundary.interpolate(i * segment_length) for i in range(max_k)]

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
                    selected_nodes[geom_key]['block_ids'] = list(selected_nodes[geom_key]['block_ids'])
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

    G_quarters = nx.Graph()
    buildings_with_blocks = assign_buildings_to_blocks(buildings, blocks)
    buildings_in_blocks = {}

    for idx, row in blocks.iterrows():
        block_id = row['block_id']
        centroid = row['centroid_node']
        x = centroid.x
        y = centroid.y
        G_quarters.add_node(f'{block_id}_block', geometry=centroid, x=x, y=y, type='centroid')
        
        buildings_in_block = buildings_with_blocks[buildings_with_blocks["block_id"] == block_id]
        buildings_in_blocks[block_id] =  buildings_in_block

    for idx, row in blocks.iterrows():
        print(idx, end=' ')
        block_id = row['block_id']
        centroid = row['centroid_node']
        
        buildings = buildings_in_blocks.get(block_id)

        connectors = connectors_gdf[connectors_gdf['block_ids'].apply(lambda x: block_id in x)]
        
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
                G_quarters.add_node(connector_id, geometry=connector_geom, x=x, y=y, type='connector', blocks=[block_node_name])
            else:
                G_quarters.nodes[connector_id]['blocks'].append(block_node_name)
            G_quarters.add_edge(block_node_name, connector_id, length_meter=mean_dist, geometry=line)
            
    G_quarters.graph['crs'] = 'EPSG:3857'

    return G_quarters


def get_blocks_graph(graph: nx.MultiDiGraph, blocks, buildings, node_merge_dist=70, connectors_count=5):
    ped_nodes, ped_edges = ox.graph_to_gdfs(graph)
    ped_nodes = ped_nodes.reset_index(drop=True)

    nodes_in_buffer = get_connector_points(blocks, ped_nodes)
    intersection_nodes = get_edge_intersection_points(blocks, ped_edges)

    buffer_nodes = nodes_in_buffer.rename(columns={'index_right': 'block_id'})
    intersection_nodes = intersection_nodes.rename(
        columns={'block_idx': 'block_id'})
    buffer_nodes = buffer_nodes.loc[:, ~buffer_nodes.columns.duplicated()]
    intersection_nodes = intersection_nodes.loc[:, ~intersection_nodes.columns.duplicated()]
    all_nodes = gpd.GeoDataFrame(
        pd.concat([buffer_nodes, intersection_nodes]),
        crs=buffer_nodes.crs
    )
    
    filtered_nodes = filter_close_nodes(all_nodes)
    merged_nodes = merge_close_nodes(
        filtered_nodes, blocks, node_merge_dist=node_merge_dist)
    final_connectors = limit_connectors_segmented(
        merged_nodes, blocks, max_k=connectors_count)
    
    blocks = blocks.to_crs(CRS)
    blocks["block_id"] = blocks.index
    blocks["centroid_node"] = blocks.geometry.representative_point()
    G_blocks = build_block_graph(blocks, buildings, graph, final_connectors, weight='length_meter')

    return G_blocks
