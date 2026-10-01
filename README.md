# overparam_reg
code for Off-Grid 2026 paper on deformable 3D registration 


<pre><code><span style="color:#000000; font-weight:bold;">🫟</span> = DiVRoC().apply <span style="color:#008000; font-style:italic;"># set up splatting</span>
sigmas = torch.linspace(<span style="color:#098658;">1</span>,<span style="color:#098658;">5</span>,<span style="color:#098658;">3</span>) <span style="color:#008000; font-style:italic;"># define smoothing scales </span>
<span style="color:#008000; font-style:italic;"># set optimisable value parameters</span>
params = torch.nn.Parameter(torch.zeros(<span style="color:#098658;">1</span>, C, K, <span style="color:#098658;">1</span>, <span style="color:#098658;">1</span>))
<span style="color:#008000; font-style:italic;"># differentiable OFF-GRID coordinate 3D points</span>
pts = ctrl_pts.reshape(<span style="color:#098658;">1</span>, -<span style="color:#098658;">1</span>, <span style="color:#098658;">1</span>, <span style="color:#098658;">1</span>, <span style="color:#098658;">3</span>)
K,C = params.shape[:<span style="color:#098658;">2</span>] <span style="color:#008000; font-style:italic;"># K = num_pts, C = num_channels</span>
cp_weights = torch.nn.Parameter(torch.ones(<span style="color:#098658;">1</span>, K, len(sigmas))\
/len(sigmas))  <span style="color:#008000; font-style:italic;"># uniform weights for each scale</span>
<span style="color:#008000; font-style:italic;"># DEFINE OPTIMISER AND TRAINING LOOP</span>
<span style="color:#008000; font-style:italic;"># keep weights positive and unit sum</span>
cp_soft = torch.softmax(cp_weights, dim=-<span style="color:#098658;">1</span>)
splat_ones = <span style="color:#000000; font-weight:bold;">🫟</span>(torch.ones_like(pts[..., :<span style="color:#098658;">1</span>]).transpose(<span style="color:#098658;">2</span>, <span style="color:#098658;">1</span>),\
 pts, (<span style="color:#098658;">1</span>, <span style="color:#098658;">1</span>, H, W, D)) <span style="color:#008000; font-style:italic;"># make values homogeneous</span>
splat_values = <span style="color:#000000; font-weight:bold;">🫟</span>(torch.tanh(params), pts, (<span style="color:#098658;">1</span>, C, H, W, D))
splat_cp_w = <span style="color:#000000; font-weight:bold;">🫟</span>(cp_soft.transpose(<span style="color:#098658;">2</span>, <span style="color:#098658;">1</span>)[...,None,None], pts,\
(<span style="color:#098658;">1</span>, len(sigmas), H, W, D))
<span style="color:#008000; font-style:italic;"># MULTI-SIGMA SMOOTHING</span>
cumulative_weights = torch.zeros(<span style="color:#098658;">1</span>,<span style="color:#098658;">1</span>,H,W,D)
cumulative_values = torch.zeros(<span style="color:#098658;">1</span>,C,H,W,D)
<span style="color:#0000FF; font-weight:bold;">for</span> i_, sigma <span style="color:#0000FF; font-weight:bold;">in</span> enumerate(sigmas):
    σ = GaussianSmoothing(sigma)
    <span style="color:#008000; font-style:italic;"># Smooth values and density</span>
    ones_smooth = σ(splat_ones)  
    values_smooth = σ(splat_values)
    <span style="color:#008000; font-style:italic;"># Smooth the specific scale weights</span>
    weights_smooth = σ(splat_cp_w[:,i_:i_+<span style="color:#098658;">1</span>,...]).div(ones_smooth+<span style="color:#098658;">1e-6</span>)
    cumulative_weights += weights_smooth
    cumulative_values += values_smooth.div(ones_smooth+<span style="color:#098658;">1e-6</span>)*weights_smooth
<span style="color:#000000; font-weight:bold;">🚀</span> = (cumulative_values/cumulative_weights.add(<span style="color:#098658;">1e-6</span>))
</code></pre>
