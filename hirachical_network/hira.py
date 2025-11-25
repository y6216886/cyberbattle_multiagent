# 下面的代码片段生成一个三层网络并返回 m.Environment（示例仅作模板，细节可按需求调整）：
def generate_hierarchical_network(n_core=2, n_dist=6, n_access=20, p_access_to_dist=0.4, p_dist_to_core=0.3):
    import networkx as nx
    from cyberbattle.simulation import model as m
    import random

    # 创建节点 id
    core_nodes = [f"core_{i}" for i in range(n_core)]
    dist_nodes = [f"dist_{i}" for i in range(n_dist)]
    access_nodes = [f"access_{i}" for i in range(n_access)]

    G = nx.DiGraph()
    # 添加所有节点占位（data 在后面填充）
    for nid in core_nodes + dist_nodes + access_nodes:
        G.add_node(nid)

    # access -> distribution edges
    for a in access_nodes:
        for d in dist_nodes:
            if random.random() < p_access_to_dist:
                G.add_edge(a, d, protocol=set(["SMB","HTTP"]))  # 协议示例，可按需要设置

    # distribution -> core edges
    for d in dist_nodes:
        for c in core_nodes:
            if random.random() < p_dist_to_core:
                G.add_edge(d, c, protocol=set(["RDP","HTTP"]))

    # 也可以加一些同层或回路边（按需）
    # 为每个节点创建 NodeInfo（参考 model.NodeInfo）
    def create_nodeinfo(node_id):
        return m.NodeInfo(
            services=[],
            value=random.randint(0,100),
            properties=[],
            vulnerabilities={},  # 可调用已有 vulnerability 生成逻辑
            firewall=m.FirewallConfiguration(),
            agent_installed=False,
            reimagable=True
        )
    # 将 node data 填回 graph
    for n in list(G.nodes):
        G.nodes[n].clear()
        G.nodes[n].update({"data": create_nodeinfo(n)})

    # 可以再调用 assign_random_labels 或自定义 vulnerability 分配
    env = m.Environment(network=G, vulnerability_library=dict([]), identifiers=m.infer_constants_from_network(G, {}))
    return env
# 如何使用

# 直接在代码中调用：
# from cyberbattle.simulation.generate_network import generate_hierarchical_network
# env = generate_hierarchical_network(...)
# 然后用 CyberBattleEnv(initial_environment=env) 或将其作为 gym 环境的初始环境。
# 可把函数包装成 new_environment_*，并在 _env 中类似 cyberbattle_random.py 调用以便训练脚本复用。
# 实现建议 / 可扩展项

# 分层特征差异化：为 core 节点设置更高 value、更多服务/更高权限漏洞；access 层更容易被入侵但价值低。
# 协议/端口分配：用边的 metadata 表示协议（已有 generate_random_traffic_network 使用类似做法）。
# 漏洞与凭证分配：借鉴 cyberbattle_model_from_traffic_graph 的密码分配 / 漏洞生成逻辑为分层场景定制。
# 可视化：用 NetworkX/Matplotlib 绘制拓扑以便调试（network = env.network; nx.draw(network, ...)）。
# 样本/测试：添加到 samples 下的一个示例文件（比如 samples/hierarchical/hierarchical.py），并在 README 或 notebook 中展示。
# 如果你愿意，我可以：

# 把上面的函数实现并直接补到仓库的 generate_network.py（或新建样例文件），
# 生成一个小的 demo 脚本，创建环境并打印/可视化网络节点统计，
# 或者把函数改写为可通过命令行传参来控制层大小和连接概率。
# 你想让我现在直接把函数实现并提交到仓库，还是先只要更详细的设计建议或示例参数配置？