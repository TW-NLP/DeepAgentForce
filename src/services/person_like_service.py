"""
用户偏好挖掘系统
1. LLM 提取对话中的实体和关系
2. NetworkX 构建知识图谱
3. 图算法（PageRank、中心性等）挖掘用户偏好
多租户支持：每个租户有独立的数据存储
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
import networkx as nx

logger = logging.getLogger(__name__)


class UserPreferenceMining():
    """用户偏好挖掘,定时任务"""

    def __init__(self, settings):
        self.settings = settings
        self.client = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL
        )
        # 🆕 多租户：每个租户有独立的知识图谱
        self._tenant_graphs: dict = {}
        self._tenant_loaded: dict = {}  # 记录是否已从文件加载

    def _get_tenant_graph(self, tenant_uuid: Optional[int] = None):
        """获取租户专属的知识图谱"""
        key = tenant_uuid if tenant_uuid is not None else "default"
        if key not in self._tenant_graphs:
            self._tenant_graphs[key] = nx.DiGraph()
            self._tenant_graphs[key].add_node('USER', type='user', label='用户')
        return self._tenant_graphs[key]

    def _get_tenant_file(self, tenant_uuid: Optional[int] = None) -> Path:
        """获取租户专属的画像文件路径"""
        key = tenant_uuid if tenant_uuid is not None else "default"
        return self.settings.PERSON_LIKE_FILE.parent / f"person_like_{key}.json"

    def _load_tenant_graph(self, tenant_uuid: Optional[int] = None):
        """从文件加载租户的知识图谱"""
        key = tenant_uuid if tenant_uuid is not None else "default"
        if self._tenant_loaded.get(key):
            return

        graph_file = self._get_tenant_file(tenant_uuid)
        if graph_file.exists():
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                G = self._get_tenant_graph(tenant_uuid)

                for node_data in data.get('graph', {}).get('nodes', []):
                    node_id = node_data.pop('id')
                    G.add_node(node_id, **node_data)

                for edge_data in data.get('graph', {}).get('edges', []):
                    src = edge_data.pop('source')
                    tgt = edge_data.pop('target')
                    G.add_edge(src, tgt, **edge_data)

                self._tenant_loaded[key] = True
                logger.info(f"📂 已从文件加载租户画像: tenant={tenant_uuid}")
            except Exception as e:
                logger.warning(f"加载画像文件失败: {e}")

    @property
    def graph(self):
        """兼容旧接口：返回 default 租户的图谱"""
        return self._get_tenant_graph(None)

    def extract_entities_relations(self, conversations: List[Dict], tenant_uuid: Optional[int] = None) -> List[Dict]:
        """步骤1: 使用 LLM 从对话中提取实体和关系"""
        conv_text = "\n".join(conversations)

        prompt = f"""从用户的对话中提取实体和关系，用于构建知识图谱。

对话记录：
{conv_text}

请提取：
1. 实体（entities）：用户提到的具体事物、概念、话题等
2. 关系（relations）：用户与实体之间的关系

以 JSON 格式返回：
{{
  "entities": [
    {{"name": "Python", "type": "技术", "mentions": 3}},
    {{"name": "数据分析", "type": "领域", "mentions": 2}}
  ],
  "relations": [
    {{"source": "USER", "target": "Python", "relation": "感兴趣", "weight": 0.9}},
    {{"source": "USER", "target": "数据分析", "relation": "想学习", "weight": 0.8}}
  ]
}}

注意：
- 实体 name 要简洁明确
- 关系 source 统一用 "USER" 代表用户
- weight 范围 0-1，表示关系强度
- 只提取明确的、有价值的实体和关系

只返回 JSON，无其他内容。"""

        print("🦀 LLM 提取实体和关系...")
        response = self.client.invoke(prompt)

        result_text = response.content
        result_text = result_text.replace('```json', '').replace('```', '').strip()

        extracted = json.loads(result_text)
        print(f"✓ 提取到 {len(extracted['entities'])} 个实体")
        print(f"✓ 提取到 {len(extracted['relations'])} 个关系")

        return extracted

    def build_knowledge_graph(self, extracted_data: Dict, tenant_uuid: Optional[int] = None):
        """步骤2: 使用 NetworkX 构建知识图谱"""
        print("\n📊 构建知识图谱...")

        G = self._get_tenant_graph(tenant_uuid)

        for entity in extracted_data['entities']:
            G.add_node(
                entity['name'],
                type='entity',
                entity_type=entity.get('type', 'unknown'),
                mentions=entity.get('mentions', 1),
                label=entity['name']
            )

        for relation in extracted_data['relations']:
            G.add_edge(
                relation['source'],
                relation['target'],
                relation=relation['relation'],
                weight=relation.get('weight', 0.5)
            )

        print(f"✓ 图谱节点数: {G.number_of_nodes()}")
        print(f"✓ 图谱边数: {G.number_of_edges()}")

    def mine_preferences_with_graph_algorithms(self, tenant_uuid: Optional[int] = None) -> Dict:
        """步骤3: 使用图算法挖掘用户偏好"""
        print("\n🔍 使用图算法挖掘偏好...")

        G = self._get_tenant_graph(tenant_uuid)

        pagerank_scores = nx.pagerank(G, weight='weight')
        entity_pagerank = {
            node: score
            for node, score in pagerank_scores.items()
            if node != 'USER'
        }
        top_pagerank = sorted(entity_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

        user_neighbors = list(G.successors('USER'))
        neighbor_weights = {}
        for neighbor in user_neighbors:
            edge_data = G['USER'][neighbor]
            neighbor_weights[neighbor] = edge_data.get('weight', 0.5)

        top_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)

        entity_mentions = {}
        entity_types = {}
        for node, data in G.nodes(data=True):
            if data.get('type') == 'entity':
                entity_mentions[node] = data.get('mentions', 0)
                entity_types[node] = data.get('entity_type', 'unknown')

        top_mentions = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:10]

        preference_scores = {}
        for entity in entity_pagerank.keys():
            score = 0.0
            score += pagerank_scores.get(entity, 0) * 40
            if entity in neighbor_weights:
                score += neighbor_weights[entity] * 30
            mentions = entity_mentions.get(entity, 0)
            max_mentions = max(entity_mentions.values()) if entity_mentions else 1
            score += (mentions / max_mentions) * 30
            preference_scores[entity] = score

        top_preferences = sorted(preference_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        result = {
            "top_preferences": [
                {
                    "entity": entity,
                    "score": round(score, 3),
                    "type": entity_types.get(entity, 'unknown'),
                    "mentions": entity_mentions.get(entity, 0),
                    "pagerank": round(pagerank_scores.get(entity, 0), 4)
                }
                for entity, score in top_preferences
            ],
            "algorithm_results": {
                "pagerank_top10": [{"entity": e, "score": round(s, 4)} for e, s in top_pagerank],
                "direct_connections": [{"entity": e, "weight": round(w, 3)} for e, w in top_neighbors],
                "most_mentioned": [{"entity": e, "mentions": m} for e, m in top_mentions]
            },
            "graph_statistics": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "user_connections": G.out_degree('USER'),
                "avg_clustering": round(nx.average_clustering(G.to_undirected()), 3)
            }
        }

        return result

    def save_graph(self, filepath: str = None, tenant_uuid: Optional[int] = None):
        """保存租户专属的知识图谱"""
        graph_file = filepath or str(self._get_tenant_file(tenant_uuid))
        G = self._get_tenant_graph(tenant_uuid)
        graph_data = {
            'nodes': [{'id': node, **data} for node, data in G.nodes(data=True)],
            'edges': [{'source': u, 'target': v, **data} for u, v, data in G.edges(data=True)]
        }
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"💾 知识图谱已保存: {graph_file}")

    def get_frontend_format(self, tenant_uuid: Optional[int] = None) -> Dict:
        """多租户：返回前端格式的用户画像"""
        self._load_tenant_graph(tenant_uuid)
        graph_file = self._get_tenant_file(tenant_uuid)
        if graph_file.exists():
            with open(graph_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def person_like_save(self, tenant_uuid: Optional[int] = None, user_id: Optional[int] = None) -> Dict:
        """多租户：保存当前租户的画像"""
        tenant_history_dir = self.settings.HISTORY_DIR / str(tenant_uuid if tenant_uuid is not None else "default")
        sessions_list = []
        if tenant_history_dir.exists():
            for file_i in os.listdir(tenant_history_dir):
                file_path = os.path.join(tenant_history_dir, file_i)
                if os.path.isfile(file_path) and file_i.startswith('session_'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        sessions_list.extend(
                            [i.get("user_content", "") for i in file_data.get('conversations', [])]
                        )
        sessions_list = [item for item in sessions_list if item.strip()]

        if not sessions_list:
            print("⚠️ 没有会话数据，跳过画像生成")
            return {}

        extracted = self.extract_entities_relations(sessions_list, tenant_uuid)
        self.build_knowledge_graph(extracted, tenant_uuid)
        G = self._get_tenant_graph(tenant_uuid)
        preferences_result = self.mine_preferences_with_graph_algorithms(tenant_uuid)

        graph_data = {
            'nodes': [
                {'id': node, 'type': data.get('type', 'unknown'), 'label': data.get('label', node)}
                for node, data in G.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, 'relation': data.get('relation', ''), 'weight': data.get('weight', 0.5)}
                for u, v, data in G.edges(data=True)
            ]
        }
        api_response = {
            'graph': graph_data,
            'preferences': preferences_result['top_preferences'],
            'statistics': preferences_result['graph_statistics'],
        }

        prompt = f"""请根据以下"用户偏好挖掘系统"的输出数据，生成一段简练、专业的【用户画像侧写】。
【输入数据】：
{json.dumps(api_response, ensure_ascii=False, indent=2)}
【要求】：
1. **核心定位**：一句话概括用户的核心身份。
2. **偏好解读**：结合 `preferences` 中的分数排名，说明用户最关注的领域或工具。
3. **关系细节**：利用 `edges` 中的 `relation` 字段来区分用户是单纯感兴趣还是有实操需求。
4. **口吻**：客观、专业。
5. **字数**：150字以内。"""
        api_response['summary'] = self.client.invoke(prompt).content.strip()
        graph_file = self._get_tenant_file(tenant_uuid)
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(api_response, f, ensure_ascii=False, indent=2)
        return api_response

    def visualize_graph(self, output_path: str = 'preference_graph.png', tenant_uuid: Optional[int] = None):
        """可视化知识图谱"""
        try:
            import matplotlib.pyplot as plt
            G = self._get_tenant_graph(tenant_uuid)
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
            entity_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'entity']
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='#FF6B6B', node_size=3000, alpha=0.9, label='用户')
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color='#4ECDC4', node_size=1500, alpha=0.8, label='实体')
            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=True, arrowsize=20, width=2)
            nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')
            plt.title(f'用户偏好知识图谱 (tenant={tenant_uuid})', fontsize=16, fontweight='bold')
            plt.legend(loc='upper left', fontsize=11)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"🎨 图谱可视化已保存: {output_path}")
        except Exception as e:
            print(f"可视化失败: {e}")
