import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from sklearn.cluster import DBSCAN
from shapely.geometry import Point

CRS = 3857


def get_centroids(blocks):
    blocks["centroid_node"] = blocks.geometry.representative_point()
    return blocks.set_geometry("centroid_node")


def get_edge_intersection_points(blocks, pedestrian_edges):
    blocks_3857 = blocks.to_crs(epsg=3857)
    ped_edges_3857 = pedestrian_edges.to_crs(epsg=3857)

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
    print(f'{len(nodes_in_blocks_buffer)} -> {len(filtered_nodes)} nodes')

    return filtered_nodes


def merge_close_nodes(nodes_gdf, blocks_gdf, node_merge_dist=2, block_boundary_dist=10):
    coords = np.array([[p.x, p.y] for p in nodes_gdf.geometry])
    clustering = DBSCAN(eps=node_merge_dist, min_samples=1).fit(coords)
    nodes_gdf['cluster'] = clustering.labels_

    merged_nodes = nodes_gdf.groupby('cluster').geometry.apply(lambda g: Point(
        np.mean([p.x for p in g]), np.mean([p.y for p in g]))).reset_index()  # pyright: ignore[reportArgumentType]

    blocks_gdf["block_id"] = blocks_gdf.index
    block_lists = []

    for pt in merged_nodes.geometry:
        close_blocks = blocks_gdf[blocks_gdf.geometry.boundary.distance(
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
    limited = []

    for block_id, block in blocks_with_centroids.iterrows():
        group = nodes_gdf[nodes_gdf['block_ids'].apply(
            lambda x: block_id in x)].copy()
        if group.empty:
            continue

        boundary = block.geometry.boundary
        segment_length = boundary.length / max_k
        segment_points = [boundary.interpolate(
            i * segment_length) for i in range(max_k)]

        selected = []

        for pt in segment_points:
            if group.empty:
                break
            distances = group.geometry.distance(pt)
            nearest_idx = distances.idxmin()
            selected.append(group.loc[nearest_idx])
            group = group.drop(nearest_idx)

        if selected:
            limited.append(pd.DataFrame(selected))

    if limited:
        return gpd.GeoDataFrame(pd.concat(limited, ignore_index=True), crs=nodes_gdf.crs)
    else:
        return gpd.GeoDataFrame(columns=nodes_gdf.columns, crs=nodes_gdf.crs)


def simpify_pedestrian_graph(graph, blocks, node_merge_dist=70, connectors_count=5):
    ped_nodes, ped_edges = ox.graph_to_gdfs(graph)
    ped_nodes = ped_nodes.reset_index(drop=True)

    nodes_in_buffer = get_connector_points(blocks, ped_nodes)
    intersection_nodes = get_edge_intersection_points(blocks, ped_edges)

    buffer_nodes = nodes_in_buffer.rename(columns={'index_right': 'block_id'})
    intersection_nodes = intersection_nodes.rename(
        columns={'block_idx': 'block_id'})
    all_nodes = gpd.GeoDataFrame(
        pd.concat([buffer_nodes, intersection_nodes], ignore_index=True),
        crs=buffer_nodes.crs
    )
    filtered_nodes = filter_close_nodes(all_nodes)
    merged_nodes = merge_close_nodes(
        filtered_nodes, blocks, node_merge_dist=node_merge_dist)
    final_connectors_segmentes = limit_connectors_segmented(
        merged_nodes, blocks, max_k=connectors_count)

    return final_connectors_segmentes
