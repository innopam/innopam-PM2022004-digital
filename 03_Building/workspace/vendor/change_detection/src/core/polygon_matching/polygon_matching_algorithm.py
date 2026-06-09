import networkx as nx
from src.core.polygon_matching import polygon_matching_utils
from src.utils import io


# index to shapefile
def indexing(poly1, poly2):
    poly1['poly1_idx'] = range(1, len(poly1) + 1)
    poly1 = poly1.reset_index(drop=True)

    poly2['poly2_idx'] = range(1, len(poly2) + 1)
    poly2 = poly2.reset_index(drop=True)

    poly1_area = poly1.geometry.area
    poly2_area = poly2.geometry.area

    poly1 = poly1.drop(columns=['area'], errors='ignore')
    idx_loc1 = poly1.columns.get_loc('poly1_idx')
    poly1.insert(loc=idx_loc1, column='area', value=poly1_area)

    poly2 = poly2.drop(columns=['area'], errors='ignore')
    idx_loc2 = poly2.columns.get_loc('poly2_idx')
    poly2.insert(loc=idx_loc2, column='area', value=poly2_area)

    outer_joined = polygon_matching_utils.outer_join(poly1, poly2, poly1_prefix="poly1", poly2_prefix="poly2")
    return poly1, poly2, outer_joined


def build_graph(joined_df):
    nodes = set()
    links = []

    for _, row in joined_df.dropna(subset=["poly1_idx", "poly2_idx"]).iterrows():
        p1 = f"p1_{int(row['poly1_idx'])}"
        p2 = f"p2_{int(row['poly2_idx'])}"
        links.append({"source": p1, "target": p2})
        nodes.update([p1, p2])

    if "poly1_idx" in joined_df.columns:
        for p1 in joined_df["poly1_idx"].dropna().unique():
            nodes.add(f"p1_{int(p1)}")
    if "poly2_idx" in joined_df.columns:
        for p2 in joined_df["poly2_idx"].dropna().unique():
            nodes.add(f"p2_{int(p2)}")

    node_list = [{"id": n} for n in sorted(nodes)]

    return {"nodes": node_list, "links": links}


def add_energy_to_links(poly1, poly2, graph_dict):
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")
    valid_links = []

    for link in graph_dict["links"]:
        p1_idx = int(link["source"].replace("p1_", ""))
        p2_idx = int(link["target"].replace("p2_", ""))

        if p1_idx not in poly1.index or p2_idx not in poly2.index:
            raise ValueError(f"Missing poly1_idx {p1_idx} or poly2_idx {p2_idx} in geometry.")

        geom1 = poly1.loc[p1_idx, "geometry"]
        geom2 = poly2.loc[p2_idx, "geometry"]

        if geom1.is_empty or geom2.is_empty:
            raise ValueError(f"Empty geometry at poly1_idx {p1_idx} or poly2_idx {p2_idx}.")

        intersection = geom1.intersection(geom2)
        union = geom1.union(geom2)

        if union.area == 0 or intersection.area == 0:
            continue

        link["energy"] = intersection.area / union.area
        valid_links.append(link)

    graph_dict["links"] = valid_links
    return graph_dict


def split_graph_by_energy(poly1, poly2, graph_dict, threshold):
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")

    G = nx.Graph()
    original_links = graph_dict["links"]
    original_nodes = graph_dict["nodes"]

    cut_links = []
    kept_links = []

    suppression = 0.7

    for link in original_links:
        energy = link.get("energy", 0)

        if energy >= threshold:
            G.add_edge(link["source"], link["target"], energy=energy)
            kept_links.append(link)
        else:
            p1_idx = int(link["source"].replace("p1_", ""))
            p2_idx = int(link["target"].replace("p2_", ""))
            geom1 = poly1.loc[p1_idx, "geometry"]
            geom2 = poly2.loc[p2_idx, "geometry"]
            area1 = poly1.loc[p1_idx, "area"]

            intersection = geom1.intersection(geom2)
            ol1 = intersection.area / area1 if area1 > 0 else 0

            if ol1 < suppression:
                cut_links.append(link)
            else:
                G.add_edge(link["source"], link["target"], energy=energy)
                kept_links.append(link)

    for node in original_nodes:
        G.add_node(node["id"])

    components_dict = {}
    new_node_list = []
    new_link_list = []

    for comp_idx, comp in enumerate(nx.connected_components(G)):
        poly1_set = sorted(int(n[3:]) for n in comp if n.startswith("p1_"))
        poly2_set = sorted(int(n[3:]) for n in comp if n.startswith("p2_"))

        components_dict[comp_idx] = {
            "poly1_set": poly1_set,
            "poly2_set": poly2_set
        }

        for n in comp:
            new_node_list.append({"id": n, "comp_idx": comp_idx})

        for u, v in G.subgraph(comp).edges:
            source, target = (u, v) if u.startswith("p1_") else (v, u)
            new_link_list.append({
                "source": source,
                "target": target,
                "comp_idx": comp_idx,
                "energy": G[source][target]["energy"]
            })

    summary = {
        "after_components": len(components_dict),
        "num_cut_links": len(cut_links)
    }
    return components_dict, {"nodes": new_node_list, "links": new_link_list}, cut_links, summary


def calculate_all_combination_metrics(poly1, poly2, components_dict, cut_links):
    poly1, poly2 = polygon_matching_utils.mark_cut_links(poly1, poly2, cut_links)
    poly1, poly2 = polygon_matching_utils.attach_metrics_from_components(components_dict, poly1, poly2)
    # combination_df = polygon_matching_utils.generate_components_df(components_dict)
    # final_metrics_df = polygon_matching_utils.compute_metrics_for_combi_df(combination_df, poly1, poly2)
    # poly1, poly2 = polygon_matching_utils.attach_metrics_to_polys(poly1, poly2, final_metrics_df)
    poly1, poly2 = polygon_matching_utils.add_component_sets_to_polys(poly1, poly2, components_dict)
    return cut_links, poly1, poly2


def algorithm_pipeline(poly1_path, poly2_path, output_path, cut_threshold):
    poly1 = io.import_shapefile(poly1_path)
    poly2 = io.import_shapefile(poly2_path)
    poly1, poly2, joined = indexing(poly1, poly2)
    graph = build_graph(joined)
    graph = add_energy_to_links(poly1, poly2, graph)
    component, graph, cut_link, summary = split_graph_by_energy(poly1, poly2, graph, cut_threshold)
    final_metrics, poly1, poly2 = calculate_all_combination_metrics(poly1, poly2, component, cut_link)
    return final_metrics, poly1, poly2
