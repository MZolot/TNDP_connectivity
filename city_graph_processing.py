import geopandas as gpd
import osmnx as ox
import networkx as nx
import neatnet

from shapely.geometry import LineString, Point

CRS = 4326


def filter_graph_elements(elements, to_keep: dict, to_loose: dict):
    filtered = elements
    for key in to_keep.keys():
        filtered = filtered[filtered[key] == to_keep[key]]
    for key in to_loose.keys():
        filtered = filtered[filtered[key] != to_loose[key]]
    return

def _filter_short_dead_end_edges(graph, length_threshold = 200):
    filtered_graph = graph.copy()
    dead_end_edges = []

    for u, v, key, data in filtered_graph.edges(keys=True, data=True):  # type: ignore
        geom = data.get("geometry")
        if geom is None:
            continue
        length = geom.length

        deg_u = filtered_graph.in_degree(u) + filtered_graph.out_degree(u)  # type: ignore
        deg_v = filtered_graph.in_degree(v) + filtered_graph.out_degree(v)  # type: ignore

        if length <= length_threshold and (deg_u == 1 or deg_v == 1):
            dead_end_edges.append((u, v, key, length))

    for u, v, key, _ in dead_end_edges:
        if filtered_graph.has_edge(u, v, key):
            filtered_graph.remove_edge(u, v, key)

    isolated_nodes = list(nx.isolates(filtered_graph))

    filtered_graph.remove_nodes_from(isolated_nodes)
    
    return filtered_graph

def _explode_multilines(gdf):
    gdf = gdf.copy()
    gdf = gdf.explode(ignore_index=True)
    gdf['geometry'] = gdf['geometry'].apply(
        lambda g: g if isinstance(g, LineString) else list(g.geoms)[0])
    return gdf


def _prepare_gdf_for_neatify(graph):
    nodes, edges = ox.graph_to_gdfs(graph)

    edges = edges.copy()
    edges = edges.join(
        nodes[["x", "y"]],
        on="u",
        rsuffix="_u"
    )
    edges = edges.join(
        nodes[["x", "y"]],
        on="v",
        rsuffix="_v"
    )

    edges["geometry"] = edges.apply(
        lambda row: row.geometry if row.geometry is not None else LineString(
            [(row.x_u, row.y_u), (row.x_v, row.y_v)]),
        axis=1
    )

    gdf = edges.copy()
    gdf = gdf.set_geometry("geometry")
    gdf = gdf.to_crs(epsg=CRS)
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())

    gdf_for_neat = _explode_multilines(gdf_utm)
    gdf_for_neat = gdf_for_neat.drop_duplicates(subset="geometry")
    
    return gdf_for_neat


def _make_graph_from_edges(edges):
    nodes_map = {}

    def get_node_id(pt):
        key = (round(pt.x, 6), round(pt.y, 6))
        if key not in nodes_map:
            nodes_map[key] = str(len(nodes_map))
        return nodes_map[key]

    u_ids = []
    v_ids = []
    for idx, row in edges.iterrows():
        u_pt = Point(row.geometry.coords[0])
        v_pt = Point(row.geometry.coords[-1])
        u_id = get_node_id(u_pt)
        v_id = get_node_id(v_pt)
        u_ids.append(u_id)
        v_ids.append(v_id)

    edges['u'] = u_ids
    edges['v'] = v_ids
    edges['key'] = range(len(edges))

    nodes_list = [{'geometry': Point(k[0], k[1]), 'x': k[0], 'y': k[1], 'osmid': v}
                  for k, v in nodes_map.items()]
    nodes = gpd.GeoDataFrame(
        nodes_list, index=[n['osmid'] for n in nodes_list], crs=edges.crs)

    edges.set_index(['u', 'v', 'key'], inplace=True)

    G = ox.graph_from_gdfs(gdf_nodes=nodes, gdf_edges=edges)
    return G


def simplify_graph(graph, dead_ends_length_threshold):
    simplified_graph = ox.simplify_graph(graph)
    gdf_for_neat = _prepare_gdf_for_neatify(simplified_graph)
    neat_edges = neatnet.neatify(gdf_for_neat).to_crs(3857)
    neatified_graph = _make_graph_from_edges(neat_edges)
    filtered_graph = _filter_short_dead_end_edges(neatified_graph, dead_ends_length_threshold)
    
    from_str = f'({len(graph.edges)} egdes, {len(graph.nodes)} nodes)'
    between_str = f'({len(neatified_graph.edges)} egdes, {len(neatified_graph.nodes)} nodes)'    
    to_str = f'({len(filtered_graph.edges)} egdes, {len(filtered_graph.nodes)} nodes)'
    print(
        f'Simplified graph: {from_str} --> {between_str} --> {to_str}')
    
    return filtered_graph
