import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

def toDisplay(x, target_dim = None):
    while(target_dim is not None and x.dim() > target_dim):
        x = x[0]
    return x.detach().cpu().numpy()

empty_axis_dict = dict(
    backgroundcolor="rgba(0, 0, 0,0)",
    gridcolor="white",
    showbackground=True,
    zerolinecolor="white",
    showticklabels = False,
)
 
empty_background_dict = dict(
    xaxis = empty_axis_dict,
    yaxis = empty_axis_dict,
    zaxis = empty_axis_dict,
    xaxis_title='',
    yaxis_title='',
    zaxis_title='',
)

def flow_traces(
    pos, flows, sizeref=1.0, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        hoverinfo = 'none',
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
        hoverinfo = 'none',
    )

    return [lines_trace, head_trace]

def visualize_correspondence(source_pts, target_pts, corr_scores, flow, 
        weights = None, skip=100):
    N_src = len(source_pts)
    N_tgt = len(target_pts)
    
    scatter_source = go.Scatter3d(
        x = source_pts[:,0], 
        y = source_pts[:,1], 
        z = source_pts[:,2], 
        mode = 'markers',
        hoverinfo = 'none',
    )

    scatter_target = go.Scatter3d(
        x = target_pts[:,0], 
        y = target_pts[:,1], 
        z = target_pts[:,2], 
        mode = 'markers',
        hoverinfo = 'none',
    )

#     scatter_target_base = go.Scatter3d(
#         x = target_pts[:,0], 
#         y = target_pts[:,1], 
#         z = target_pts[:,2], 
#         mode = 'markers',
#         hoverinfo = 'none',
#     )
    
    lines_flow, scatter_flow = flow_traces(source_pts[::skip], flow[::skip])
    lines_flow_selected, scatter_flow_selected = flow_traces(source_pts[:1], flow[:1])

    fig = go.FigureWidget(
        make_subplots(
            column_widths=[0.5, 0.25, 0.25],
            row_heights=[1],#[0.5, 0.5],
            rows=1, cols=3,
            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
            # specs=[[{'type': 'surface', "rowspan": 2}, {'type': 'surface'}],[None, {'type': 'surface'}]],
        )
    )
    
    fig.add_trace(scatter_source, row=1,col=2)
    fig.add_trace(scatter_target, row=1,col=3)
    fig.add_trace(scatter_target, row=1,col=3)

    fig.add_trace(scatter_source, row=1,col=1)
    fig.add_trace(scatter_target, row=1,col=1)
    
    fig.add_trace(lines_flow, row=1,col=1)
    fig.add_trace(scatter_flow, row=1,col=1)

    fig.add_trace(lines_flow_selected, row=1,col=1)
    fig.add_trace(scatter_flow_selected, row=1,col=1)

    target_colors_joint = ['green'] * N_tgt
    source_colors_joint = ['#a3a7e4'] * N_src
    source_sizes_joint = [2] * N_src
    # source_sizes = [5] * N_src

    if(weights is not None):
        source_sizes = 20 * weights + 10
        weights_colors = weights
    else:
        source_sizes = [5] * N_src
        weights_colors = ['#a3a7e4'] * N_src

    scatter_source_joint = fig.data[3]
    scatter_source_joint.marker.size = source_sizes_joint
    # scatter_source_joint.marker.opacity = 0.5
    scatter_source_joint.marker.color = source_colors_joint
    scatter_source_joint.marker.line.width = 0

    scatter_target_joint = fig.data[4]
    scatter_target_joint.marker.size = [2] * N_tgt
    # scatter_target_joint.marker.opacity = 0.5
    scatter_target_joint.marker.color = target_colors_joint
    scatter_target_joint.marker.line.width = 0

    line_flow = fig.data[5]
    line_flow.line.color = 'rgba(255, 0, 0, 0.5)'
    line_flow.line.width = 1
    scatter_flow = fig.data[6]
    scatter_flow.marker.line.width = 0
    scatter_flow.marker.size = 2

    line_flow_selected = fig.data[7]
    scatter_flow_selected = fig.data[8]

    line_flow_selected.line.color = 'rgba(255, 0, 0, 0.5)'
    line_flow_selected.line.width = 0
    scatter_flow_selected.marker.line.width = 0
    scatter_flow_selected.marker.size = 0
    
    scatter_source = fig.data[0]
    scatter_target = fig.data[1]
    scatter_target_base = fig.data[2]

    scatter_source.marker.color = weights_colors
    scatter_source.marker.size = source_sizes
    scatter_source.marker.opacity = 0.5
    # scatter_source.marker.line.color = 0
    scatter_source.marker.line.width = 0

    scatter_target.marker.color = 'gray'
    scatter_target.marker.size = [5] * N_tgt
    scatter_target.marker.opacity = 0.0
    # scatter_target.marker.line.color = 0
    scatter_target.marker.line.width = 0

    scatter_target_base.marker.color = 'gray'
    scatter_target_base.marker.size = [5] * N_tgt
    scatter_target_base.marker.opacity = 0.1
    # scatter_target_base.marker.line.color = 0
    scatter_target_base.marker.line.width = 0
    # scatter_target_base.marker.size = 3

    fig.layout.hovermode = 'closest'
    
    def click_callback(trace, points, selector):
        c_src = weights_colors.copy()
        s_src = source_sizes.copy()
        for i in points.point_inds:
            # c_src[i] = 'red'
            s = corr_scores[i] / np.max(corr_scores[i])
            c_tgt = s
            s_tgt = s*20
            s_src[i] = 20
            
            flow_start = source_pts[i]
            flow_end = flow_start + flow[i]
            
            with fig.batch_update():
                scatter_source.marker.color = c_src
                scatter_target.marker.color = c_tgt
                # scatter_source_joint.marker.color = c_src
                scatter_source.marker.size = s_src
                scatter_target.marker.size = s_tgt
                # scatter_source_joint.marker.size = s_src
                
                line_flow_selected.x = (flow_start[0], flow_end[0], None)
                line_flow_selected.y = (flow_start[1], flow_end[1], None)
                line_flow_selected.z = (flow_start[2], flow_end[2], None)
                
                scatter_flow_selected.x = (flow_start[0], flow_end[0],)
                scatter_flow_selected.y = (flow_start[1], flow_end[1],)
                scatter_flow_selected.z = (flow_start[2], flow_end[2],)
                
                line_flow_selected.line.color = 'rgba(255, 0, 0, 0.5)'
                line_flow_selected.line.width = 5
                scatter_flow_selected.marker.line.width = 0
                scatter_flow_selected.marker.size = 5

    # def hover_callback(trace, points, selector):
    #     c = list(scatter.marker.color)
    #     s = list(scatter.marker.size)
    #     for i in points.point_inds:
    #         c[i] = 'red'
    #         with fig.batch_update():
    #             scatter.marker.color = c
    #             scatter.marker.size = s

    scatter_source.on_click(click_callback)
    # scatter_source.on_hover(click_callback)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        scene = empty_background_dict,
        scene2 = empty_background_dict,
        scene3 = empty_background_dict,
    )
    return fig
