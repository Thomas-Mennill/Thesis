from graphviz import Digraph

methodology_graph = Digraph("methodology", format="pdf")

methodology_graph.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)

methodology_graph.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

methodology_graph.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)


redefine_nodes = Digraph("methodology", format="pdf")

redefine_nodes.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)

redefine_nodes.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

redefine_nodes.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)

splitter = Digraph("methodology", format="pdf")

splitter.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)
    
splitter.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

splitter.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)

split_network = Digraph("methodology", format="pdf")

split_network.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)

split_network.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

split_network.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)

reassemble_network = Digraph("methodology", format="pdf")

reassemble_network.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)

reassemble_network.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

reassemble_network.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)

route_construction = Digraph("methodology", format="pdf")

route_construction.attr(
    "graph",
    rankdir="TB",
    fontname="Arial",
    fontsize="10",
    bgcolor="white",
    pad="0.2",
    nodesep="0.35",
    ranksep="0.5",
)

route_construction.attr(
    "node",
    shape="box",
    style="rounded",
    fontname="Arial",
    fontsize="10",
    color="#333333",
    penwidth="1.2",
    margin="0.12,0.08",
)

route_construction.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
    color="#555555",
    arrowsize="0.7",
)

#Split Network into Sections
methodology_graph.node("ORM_Speeds", "Speed Limits", shape="ellipse")
methodology_graph.node("ORM_Platforms", "Platform Locations", shape="ellipse")
methodology_graph.node("ORM_Geometry", "Railway Geometry", shape="ellipse")
methodology_graph.node("ORM_Signals", "Signal Locations", shape="ellipse")
methodology_graph.node("ORM_Data", "Open Railway Map Dataset", shape="ellipse")
methodology_graph.node("ORM_Noise_Removal", "Remove unwanted data", shape="diamond")
methodology_graph.node("ORM_Node_Redefinition", "Redefine railway network nodes", shape="square")
methodology_graph.node("ORM_ID_Definition", "Assign a unique, unchanging ID to each edge and node", shape="diamond")
methodology_graph.node("ORM_Node_Edge_Assignment", "Assign edges and nodes to stations", shape="diamond")
methodology_graph.node("ORM_Splitting", "Split railway network into stations and sections", shape="diamond")
methodology_graph.node("stations", "A set of station subgraphs", shape="ellipse")
methodology_graph.node("sections", "A set of section subgraphs", shape="ellipse")

#Redefine Nodes sub-process
redefine_nodes.node("noiseless_data", "Preprocessed Data", shape="ellipse")
redefine_nodes.node("classify_nodes", "Classify Existing Railway Nodes", shape="diamond")
redefine_nodes.node("find_platform_edges", "Find platform edges", shape="diamond")
redefine_nodes.node("find_run_through_edges", "Find the ends of run-through tracks", shape="diamond")
redefine_nodes.node("remove_extra_nodes", "Remove extra nodes", shape="diamond")
redefine_nodes.node("redefined_nodes", "Redefined Railway Nodes", shape="ellipse")
redefine_nodes.edge("noiseless_data", "classify_nodes")
redefine_nodes.edge("classify_nodes", "find_platform_edges")
redefine_nodes.edge("find_platform_edges", "find_run_through_edges")
redefine_nodes.edge("find_run_through_edges", "remove_extra_nodes")
redefine_nodes.edge("remove_extra_nodes", "redefined_nodes")


#Reassemble Network
methodology_graph.node("ORM_Reassembly", "Reassemble the desired railway network", shape="diamond")
methodology_graph.node("reassembled_network", "Reassembled Railway Network", shape="ellipse")


methodology_graph.node("running_time_calculator", "Minimum Running Time Calculator", shape="diamond")
methodology_graph.node("optimization_constraints", "Optimization Constraints", shape="ellipse")
methodology_graph.node("running_time_results", "Minimum Running Times", shape="ellipse")
methodology_graph.node("optimization_constraints", "Optimization Constraints", shape="ellipse")
methodology_graph.node("optimization_model", "Optimization Model", shape="diamond")
methodology_graph.node("optimized_timetable", "Optimized Timetable", shape="ellipse")



methodology_graph.edge("running_time_calculator", "running_time_results")
methodology_graph.edge("running_time_results", "optimization_model")
methodology_graph.edge("optimization_constraints", "optimization_model")
methodology_graph.edge("optimization_model", "optimized_timetable")

methodology_graph.render("methodology_flowchart", cleanup=True)